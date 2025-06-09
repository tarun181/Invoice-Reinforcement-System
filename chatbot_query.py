import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_together import Together
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=TOGETHER_API_KEY, max_tokens=256)

prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent HR assistant helping employees with their reimbursement queries in India.
Use the retrieved documents to answer the user's question clearly, helpfully, and always in Indian context.

- All travel locations, expenses, and reimbursements are related to India.
- All currency should be assumed and shown in INR ₹ (Indian Rupees), not USD.
- Avoid references to foreign locations (like USA) unless they explicitly appear in the context.
- Format your answers in **markdown** for readability.

Question: {query}

Context:
{context}
""")


chat_history = {}


def get_session_history(session_id: str):
    if session_id not in chat_history:
        chat_history[session_id] = InMemoryChatMessageHistory()
    return chat_history[session_id]


def build_chain():
    return RunnableWithMessageHistory(
        LLMChain(llm=llm, prompt=prompt_template),
        lambda session_id: get_session_history(session_id),
        input_messages_key="query"
    )


rag_chain = build_chain()


def retrieve_context(query: str, metadata: dict = None, k: int = 2) -> str:
    if metadata:
        results = vector_db.similarity_search(query, k=k, filter=metadata)
    else:
        results = vector_db.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in results])


def extract_metadata_filters(query: str) -> tuple[str, dict]:
    metadata = {}
    cleaned_query = query.strip()

    if "employee:" in cleaned_query:
        parts = cleaned_query.split("employee:")
        cleaned_query = parts[0].strip()
        metadata["employee_name"] = parts[1].strip()

    if "status:" in cleaned_query:
        parts = cleaned_query.split("status:")
        cleaned_query = parts[0].strip()
        metadata["status"] = parts[1].strip().capitalize()

    if "date:" in cleaned_query:
        parts = cleaned_query.split("date:")
        cleaned_query = parts[0].strip()
        metadata["date"] = parts[1].strip()

    return cleaned_query, metadata


def answer_query(user_query: str, session_id: str = "default-session") -> str:
    query, metadata = extract_metadata_filters(user_query)
    context = retrieve_context(query, metadata)
    response = rag_chain.invoke({"query": query, "context": context}, config={"configurable": {"session_id": session_id}})
    return response["text"]
