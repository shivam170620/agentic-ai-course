from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command
from typing_extensions import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os, arxiv, requests
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


supervisor_agent_prompt = """
You are a supervisor agent that coordinates the workflow of multiple agents.
Your task is to decide which agent to invoke next based on the current state of the workflow.

You will receive a state object containing:
- Current messages exchanged so far
- Results of previously executed agents

The available agents are:
- search_agent: Performs a search based on the query.
- data_agent: Processes the search results and prepares data for analysis.
- nlp_agent: Analyzes the data and extracts insights.
- report_agent: Generates a report based on the insights extracted by the nlp_agent.

Decision rules:
1. If search results are available → invoke `data_agent`.
2. If processed data is available → invoke `nlp_agent`.
3. If insights from `nlp_agent` are available → invoke `report_agent`.
4. If the report is generated → respond with `"next": "__end__"`.
5. If none of the above conditions are met, invoke `supervisor_agent` again.

When the overall workflow is complete, respond with `"__end__"`.

Return ONLY in the following JSON format:
{
   "next": <member_id>
}
"""

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
)

def decide_sources(query: str) -> list:
    prompt = f"""
    You are a source selection assistant. 
    Decide which of these sources to use for the query: Wikipedia, arXiv, SemanticScholar, Crossref, Pubmed.
    Return only a Python list of source names in lowercase.
    Query: "{query}"
    """
    response = llm.invoke(prompt)
    return eval(response.content)  

def search_arxiv(query, max_results=5):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return [{
        "title": r.title,
        "summary": r.summary,
        "url": r.entry_id,
        "published": r.published.strftime("%Y-%m-%d"),
        "source": "arxiv"
    } for r in search.results()]

def search_semantic_scholar(query, limit=5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,url,abstract,authors,year"}
    r = requests.get(url, params=params).json()
    return [{
        "title": p.get("title"),
        "summary": p.get("abstract"),
        "url": p.get("url"),
        "published": p.get("year"),
        "source": "semantic_scholar"
    } for p in r.get("data", [])]

def search_pubmed(query, max_results=5):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    ids = requests.get(search_url).json().get("esearchresult", {}).get("idlist", [])
    results = []
    for pmid in ids:
        summary_url = f"{base}esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
        data = requests.get(summary_url).json().get("result", {}).get(pmid, {})
        results.append({
            "title": data.get("title"),
            "summary": None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "published": data.get("pubdate"),
            "source": "pubmed"
        })
    return results

def search_crossref(query, max_results=5):
    url = f"https://api.crossref.org/works?query={query}&rows={max_results}"
    r = requests.get(url).json()
    return [{
        "title": i.get("title", [""])[0],
        "summary": None,
        "url": i.get("URL"),
        "published": i.get("created", {}).get("date-time", ""),
        "source": "crossref"
    } for i in r.get("message", {}).get("items", [])]

def search_wikipedia(query, limit=5):
    """
    Search Wikipedia for a given query using the Wikipedia REST API.

    Args:
        query (str): Search query string.
        limit (int): Number of results to return (default is 5).

    Returns:
        list[dict]: List of search results with title, snippet, and URL.
    """
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": limit
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            snippet = item["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append({
                "title": title,
                "snippet": snippet,
                "url": page_url
            })

        return results

    except Exception as e:
        print(f"[ERROR] Wikipedia search failed: {e}")
        return []

# -------------------------
# AGENT IMPLEMENTATIONS
# -------------------------
def supervisor_agent(state: MessagesState) -> Command[Literal["search_agent","data_agent", "nlp_agent", "report_agent"]]:
    """
    Supervisor agent that coordinates the workflow.
    It decides which agent to invoke next based on the state.
    """

    # Based on the state messages, determine which agent to invoke next, .have overall messages and decide the next step also 
    # in state["messages"] we have the last messgaes as the human messages

    messages = state["messages"] + {"role": "user", "content" : supervisor_agent_prompt}
    response = llm.invoke(messages)
    goto = response["next"]
    return Command(update=goto ,update={"messages": state["messages"]})

def search_agent(state: MessagesState) -> Command[Literal["supervisor_agent"]]:
    """
    Search agent that performs a search based on the query.
    It updates the state with the search results.
    """
    query = state["messages"][-1].content  # Get the last user message as the query
    selected_sources = decide_sources(query)
    results = []

    if "wikipedia" in selected_sources:
        results.extend(search_wikipedia(query))
    if "arxiv" in selected_sources:
        results.extend(search_arxiv(query))
    if "semantic_scholar" in selected_sources:
        results.extend(search_semantic_scholar(query))
    if "crossref" in selected_sources:
        results.extend(search_crossref(query))
    if "pubmed" in selected_sources:
        results.extend(search_pubmed(query))

    seen = set()
    deduped = []
    for r in results:
        if r["title"] and r["title"].lower() not in seen:
            deduped.append(r)
            seen.add(r["title"].lower())

    state["search_results"] = deduped
    state["search_done"] = True
    print(f"[SearchAgent] Found {len(deduped)} unique papers for '{query}'.")
    return Command(goto = "supervisor_agent", update={ "messages" : state["messages"] + [deduped]})

def data_agent(state: MessagesState) -> Command[Literal["supervisor_agent"]]:
    """
    Data Agent:
    - Cleans and filters search results.
    - Summarizes relevant papers.
    - Passes summaries back to the supervisor for later report generation.
    """
    print("[DataAgent] Cleaning, filtering, and summarizing...")

    query = state["messages"][-1].content if state["messages"] else ""
    search_results = state.get("search_results", [])

    # 1. Cleaning: remove entries without title/abstract or too short
    cleaned = []
    for paper in search_results:
        title = paper.get("title", "").strip()
        snippet = paper.get("snippet") or paper.get("summary") or ""
        snippet = snippet.strip()

        if title and len(snippet) > 50:  # arbitrary length filter
            cleaned.append({"title": title, "snippet": snippet, "url": paper.get("url")})

    # 2. Summarization: use LLM for each cleaned paper
    summaries = []
    for paper in cleaned:
        prompt = f"""
        Summarize the following research document in 3-4 sentences for a researcher interested in:
        '{query}'.
        Title: {paper['title']}
        Abstract/Snippet: {paper['snippet']}
        """
        try:
            llm_response = llm.invoke(prompt)  # Replace with your LLM call
            summary_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        except Exception as e:
            print(f"[DataAgent] Summarization failed for '{paper['title']}': {e}")
            summary_text = paper['snippet']

        summaries.append({
            "title": paper["title"],
            "summary": summary_text,
            "url": paper.get("url")
        })

    # 3. Store results in state
    state["cleaned_papers"] = cleaned
    state["summaries"] = summaries

    print(f"[DataAgent] Processed {len(cleaned)} papers. Summaries ready.")
    return Command(
        goto="supervisor_agent",
        update={
            "messages": state["messages"] + [{"role": "assistant", "content": f"Summarized {len(cleaned)} papers"}],
            "cleaned_papers": cleaned,
            "summaries": summaries
        }
    )

def report_agent(state) -> Command[Literal["__end__"]]:
    """Report Agent compiles final report into PDF."""
    print("[ReportAgent] Compiling report...")

    # Extract data from state
    summaries = state.get("summaries", [])
    query = state.get("query", "research")
    data_agent_info = state.get("data_agent_data", {})  # from data_agent

    # Create report text
    report_text = "\n".join(summaries)
    final_report = f"Research Report for query: {query}\n\n"
    final_report += "=== Summary ===\n" + report_text + "\n\n"
    final_report += "=== Data Collected ===\n"
    for key, value in data_agent_info.items():
        final_report += f"{key}: {value}\n"

    state["final_report"] = final_report

    # Ensure reports folder exists
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # File name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{query}_{timestamp}.pdf"
    file_path = os.path.join(reports_dir, filename)

    # PDF styles
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Research Report for: {query}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Summary:</b>", styles['Heading2']))
    story.append(Paragraph(report_text.replace("\n", "<br/>"), styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Data Collected:</b>", styles['Heading2']))

    for key, value in data_agent_info.items():
        story.append(Paragraph(f"<b>{key}</b>: {value}", styles['Normal']))
        story.append(Spacer(1, 6))

    # Build PDF
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    doc.build(story)

    print(f"[ReportAgent] Report saved at {file_path}")
    print("\n=== FINAL OUTPUT ===")
    print(state["final_report"])

    return Command(goto="__end__", update=state)




