from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv

from analyze_invoices import process_invoices
from chatbot_query import answer_query

# --- Initialize FastAPI app ---
app = FastAPI()
load_dotenv()

# --- File validation settings ---
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_ZIP_EXTENSIONS = {".zip", ".rar", ".7z"}

# --- Utilities ---
def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions

def create_safe_filename(filename: str) -> str:
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    return "".join(c for c in filename if c in safe_chars)

# --- Upload endpoint ---
@app.post("/upload/")
async def upload_files(
    employee_name: str = Form(...),
    policy_pdf: UploadFile = File(...),
    invoices_zip: UploadFile = File(...)
):
    try:
        # Validate inputs
        if not validate_file_extension(policy_pdf.filename, ALLOWED_PDF_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Policy file must be a PDF")

        if not validate_file_extension(invoices_zip.filename, ALLOWED_ZIP_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invoices file must be a ZIP, RAR, or 7Z archive")

        # Save files
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        safe_policy = create_safe_filename(policy_pdf.filename)
        safe_zip = create_safe_filename(invoices_zip.filename)
        policy_path = data_dir / safe_policy
        invoices_path = data_dir / safe_zip

        with open(policy_path, "wb") as f:
            shutil.copyfileobj(policy_pdf.file, f)

        with open(invoices_path, "wb") as f:
            shutil.copyfileobj(invoices_zip.file, f)

        # Process invoices
        results = process_invoices(str(policy_path), str(invoices_path), employee_name.strip())

        return {
            "success": True,
            "status": "completed",
            "employee_name": employee_name.strip(),
            "files_processed": {
                "policy_pdf": safe_policy,
                "invoices_zip": safe_zip
            },
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Query endpoint ---
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default-session"

@app.post("/query/")
async def query_rag(request: QueryRequest):
    try:
        answer = answer_query(request.query, request.session_id)
        return {
            "success": True,
            "response": answer
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# --- Health check ---
@app.get("/health")
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "message": "Service is running"
    }

# --- Root metadata ---
@app.get("/")
async def root():
    return {
        "success": True,
        "message": "Invoice Reimbursement System API",
        "endpoints": {
            "upload": "/upload/",
            "query": "/query/",
            "health": "/health"
        }
    }
