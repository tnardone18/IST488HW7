__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import anthropic

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="News Information Bot", page_icon="📰")
st.title("📰 News Information Bot")
st.write(
    "A news monitoring bot for a global law firm. "
    "Ask about specific topics, companies, or request the most interesting news."
)

# ─────────────────────────────────────────────
# API KEYS (sidebar)
# ─────────────────────────────────────────────
openai_api_key = st.secrets["OPENAI_API_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

with st.sidebar:
    st.header("⚙️ Configuration")

    model_choice = st.radio(
        "Select LLM",
        ["gpt-4o-mini (Lower Cost)", "claude-sonnet-4-20250514 (Higher Cost)"],
        index=0,
    )

    st.divider()
    st.markdown("**Sample queries:**")
    st.markdown("- Find the most interesting news")
    st.markdown("- Find news about Microsoft")
    st.markdown("- What legal risks are in the latest articles?")

# ─────────────────────────────────────────────
# LOAD & BUILD CHROMA DB (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def build_chroma_db(api_key):
    """Load articles from CSV and build a ChromaDB collection."""
    df = pd.read_csv("news.csv")

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        doc_text = (
            f"Company: {row['company_name']}\n"
            f"Date: {row['Date']}\n"
            f"Article: {row['Document']}\n"
            f"URL: {row['URL']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "company_name": str(row["company_name"]),
            "date": str(row["Date"]),
            "url": str(row["URL"]),
        })
        ids.append(f"article_{i}")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="news_articles",
        embedding_function=openai_ef,
    )

    if collection.count() == 0:
        # ChromaDB has a batch limit; add in chunks
        BATCH = 500
        for start in range(0, len(documents), BATCH):
            end = start + BATCH
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    return collection, df


try:
    collection, df = build_chroma_db(openai_api_key)
    st.sidebar.success(f"✅ Loaded {collection.count()} articles")
except FileNotFoundError:
    st.error("❌ `news.csv` not found. Place your CSV file in the app directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# TOOL DEFINITIONS (used by both OpenAI and Anthropic)
# ─────────────────────────────────────────────

# --- Tool implementation functions ---

def search_news_by_topic(query: str, n_results: int = 5) -> str:
    """Semantic search across all articles."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )
    return _format_results(results)


def search_news_by_company(company_name: str, n_results: int = 5) -> str:
    """Search for articles about a specific company using metadata filter."""
    # First try exact metadata match
    try:
        results = collection.query(
            query_texts=[company_name],
            where={"company_name": {"$eq": company_name}},
            n_results=min(n_results, collection.count()),
        )
        if results["documents"][0]:
            return _format_results(results)
    except Exception:
        pass

    # Fallback: semantic search with company name
    results = collection.query(
        query_texts=[f"news about {company_name}"],
        n_results=min(n_results, collection.count()),
    )
    return _format_results(results)


def get_interesting_news(n_results: int = 10) -> str:
    """Retrieve a broad set of articles for the LLM to rank by interest."""
    queries = [
        "major lawsuit litigation legal action",
        "merger acquisition deal partnership",
        "regulatory investigation fine penalty",
        "earnings revenue financial results",
        "executive leadership change resignation",
    ]
    seen_ids = set()
    all_docs = []
    all_meta = []

    for q in queries:
        results = collection.query(
            query_texts=[q],
            n_results=min(5, collection.count()),
        )
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(results["documents"][0][i])
                all_meta.append(results["metadatas"][0][i])

    # Format the combined results
    parts = []
    for i, (doc, meta) in enumerate(zip(all_docs, all_meta)):
        parts.append(
            f"--- Article {i+1} ---\n"
            f"Company: {meta.get('company_name', 'N/A')}\n"
            f"Date: {meta.get('date', 'N/A')}\n"
            f"{doc}\n"
            f"URL: {meta.get('url', 'N/A')}"
        )
    return "\n\n".join(parts) if parts else "No articles found."


def _format_results(results) -> str:
    """Format ChromaDB query results into readable context."""
    parts = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        parts.append(
            f"--- Article {i+1} ---\n"
            f"Company: {meta.get('company_name', 'N/A')}\n"
            f"Date: {meta.get('date', 'N/A')}\n"
            f"{doc}\n"
            f"URL: {meta.get('url', 'N/A')}"
        )
    return "\n\n".join(parts) if parts else "No articles found."


# --- Tool schemas for the LLMs ---

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news_by_topic",
            "description": "Search news articles by topic, keyword, or general query. Use this when the user asks about a broad subject like 'technology', 'legal risks', 'regulations', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or topic to look for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news_by_company",
            "description": "Search news articles about a specific company. Use this when the user mentions a company name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to search for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_interesting_news",
            "description": "Retrieve a broad set of potentially interesting and impactful news articles for the user to review. Use this when the user asks for 'interesting news', 'top stories', 'most important news', or similar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
]

ANTHROPIC_TOOLS = [
    {
        "name": "search_news_by_topic",
        "description": "Search news articles by topic, keyword, or general query. Use this when the user asks about a broad subject like 'technology', 'legal risks', 'regulations', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic to look for",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_news_by_company",
        "description": "Search news articles about a specific company. Use this when the user mentions a company name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "The company name to search for",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "get_interesting_news",
        "description": "Retrieve a broad set of potentially interesting and impactful news articles for the user to review. Use this when the user asks for 'interesting news', 'top stories', 'most important news', or similar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "search_news_by_topic": search_news_by_topic,
    "search_news_by_company": search_news_by_company,
    "get_interesting_news": get_interesting_news,
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a news monitoring assistant for a large global law firm.
Your job is to help attorneys and staff find and understand news relevant to their clients.

You ONLY report on news from the articles retrieved by your tools. Do not make up or 
reference any news stories not returned by the tools.

You have three tools available:
1. search_news_by_topic — for general topic or keyword searches
2. search_news_by_company — for company-specific searches
3. get_interesting_news — for retrieving a broad set of high-impact articles

ALWAYS use at least one tool before answering. Choose the most appropriate tool based 
on the user's query.

When ranking "interesting" news, consider these factors for a law firm audience:
- Litigation risk or active lawsuits
- Regulatory actions, fines, or investigations
- Mergers, acquisitions, or major deals
- Executive changes or corporate governance issues
- Market-moving financial events
- Reputational risk or PR crises

Present ranked results as a numbered list. For each article include:
1. A brief headline/summary
2. The company name
3. Why it matters (legal/business impact)
4. The source URL

If no relevant articles are found, say so clearly — do not fabricate information."""

# ─────────────────────────────────────────────
# LLM RESPONSE WITH TOOL CALLING
# ─────────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool function by name with given arguments."""
    func = TOOL_FUNCTIONS.get(name)
    if func:
        return func(**arguments)
    return "Tool not found."


def get_openai_response(messages):
    """Handle OpenAI chat with tool calling loop."""
    client = OpenAI(api_key=openai_api_key)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # First call — may request tool use
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_messages,
        tools=OPENAI_TOOLS,
    )

    msg = response.choices[0].message

    # Tool calling loop (allow up to 3 rounds)
    rounds = 0
    while msg.tool_calls and rounds < 3:
        rounds += 1
        api_messages.append(msg)

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, args)
            api_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            tools=OPENAI_TOOLS,
        )
        msg = response.choices[0].message

    return msg.content or "I couldn't generate a response."


def get_anthropic_response(messages):
    """Handle Anthropic chat with tool calling loop."""
    if not anthropic_api_key:
        return None

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    api_messages = []
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # First call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=ANTHROPIC_TOOLS,
        messages=api_messages,
    )

    # Tool calling loop (allow up to 3 rounds)
    rounds = 0
    while response.stop_reason == "tool_use" and rounds < 3:
        rounds += 1

        # Collect tool use blocks and execute them
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Send results back
        api_messages.append({"role": "assistant", "content": response.content})
        api_messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOLS,
            messages=api_messages,
        )

    # Extract final text
    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    return "\n".join(text_parts) if text_parts else "I couldn't generate a response."


# ─────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the news..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching articles and generating response..."):
            if "gpt" in model_choice:
                response = get_openai_response(st.session_state.messages)
            else:
                response = get_anthropic_response(st.session_state.messages)
                if response is None:
                    response = "⚠️ Anthropic API key not found in secrets."

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})