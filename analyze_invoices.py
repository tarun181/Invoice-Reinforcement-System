import os
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_together import Together
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)


def load_policy_text(policy_pdf_path):
    loader = PyMuPDFLoader(policy_pdf_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])


def extract_invoice_paths(zip_path, extract_to="invoices"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        return [os.path.join(extract_to, f) for f in zip_ref.namelist() if f.endswith(".pdf")]


def setup_llm_and_prompt():
    llm = Together(model=LLM_MODEL, api_key=TOGETHER_API_KEY, max_tokens=256)

    status_schema = ResponseSchema(name="status", description="Reimbursement decision.")
    reason_schema = ResponseSchema(name="reason", description="Reason for reimbursement decision.")
    output_parser = StructuredOutputParser.from_response_schemas([status_schema, reason_schema])
    format_instructions = output_parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_template(f"""
    You are an HR assistant analyzing employee reimbursement claims.
    Categorize the invoice as Fully Reimbursed, Partially Reimbursed, or Declined.
    Give a clear reason based on policy.

    Policy:
    {{policy}}

    Invoice:
    {{invoice}}

    {format_instructions}
    """)

    return llm, prompt_template, output_parser


def analyze_invoice(invoice_path, policy_text, employee_name, vector_store, llm, prompt, output_parser):
    loader = PyMuPDFLoader(invoice_path)
    docs = loader.load()
    invoice_text = "\n".join([doc.page_content for doc in docs])

    messages = prompt.format_messages(
        policy=policy_text,
        invoice=invoice_text,
        format_instructions=output_parser.get_format_instructions()
    )

    response = llm.invoke(messages)
    result = output_parser.parse(response)

    metadata = {
        "invoice_id": os.path.basename(invoice_path),
        "employee_name": employee_name,
        "status": result["status"],
        "reason": result["reason"],
        "date": datetime.now().isoformat()
    }

    full_text = invoice_text + "\n\nLLM Result:\n" + result["reason"]
    chunks = TEXT_SPLITTER.split_text(full_text)
    vector_store.add_texts(chunks, metadatas=[metadata] * len(chunks))
    return metadata


def process_invoices(policy_pdf_path, invoices_zip_path, employee_name):
    policy_text = load_policy_text(policy_pdf_path)
    invoice_paths = extract_invoice_paths(invoices_zip_path)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    llm, prompt, output_parser = setup_llm_and_prompt()

    results = []
    for path in invoice_paths:
        try:
            results.append(analyze_invoice(path, policy_text, employee_name, vector_store, llm, prompt, output_parser))
        except Exception as e:
            results.append({"invoice_id": os.path.basename(path), "error": str(e)})
    return results
