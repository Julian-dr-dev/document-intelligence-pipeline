import os
import shutil
import tempfile
from typing import List, Dict


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from pipeline import process_document
from extractor import load_extractor, extract_from_page



app = FastAPI(
    title="Document Intelligence API",
    description="Extract structured fields from PDFs using LayoutLMv2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("[app] Loading models at startup...")
processor, model = load_extractor()
print("[app] Models ready.")




class ExtractedFields(BaseModel):
    vendor:      str
    date:        str
    invoice_num: str
    total:       str
    address:     str
 
 
class LabeledWord(BaseModel):
    word:  str
    label: str
    box:   List[int]
 
 
class PageResult(BaseModel):
    page:          int
    fields:        ExtractedFields
    labeled_words: List[LabeledWord]
 
 
class ExtractionResponse(BaseModel):
    filename: str
    pages:    List[PageResult]
    status:   str
 
 
class HealthResponse(BaseModel):
    status:  str
    model:   str
    device:  str









@app.get("/api/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns the model name and device being used.
    """
    return HealthResponse(
        status="ok",
        model="LayoutLMv2",
        device="cuda" if str(next(model.parameters()).device) != "cpu" else "cpu",
    )
 
 
@app.post("/api/extract", response_model=ExtractionResponse)
async def extract(file: UploadFile = File(...)):
   
 
    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a PDF. Got: {file.filename}"
        )
 
    
    tmp_path = None
    try:
        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            prefix="doc_intel_"
        ) as tmp:
            # Copy the uploaded file stream into the temp file
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
 
        print(f"[app] Saved upload to temp file: {tmp_path}")
 
        # ── Run the pipeline ──────────────────────────────────────────────────
        print(f"[app] Running document pipeline...")
        pages = process_document(tmp_path)
 
        # ── Run extraction on each page ───────────────────────────────────────
        print(f"[app] Running extraction on {len(pages)} page(s)...")
        page_results = []
 
        for page in pages:
            result = extract_from_page(processor, model, page)
 
            page_results.append(PageResult(
                page=result["page"],
                fields=ExtractedFields(**result["fields"]),
                labeled_words=[
                    LabeledWord(**w) for w in result["labeled_words"]
                ],
            ))
 
        return ExtractionResponse(
            filename=file.filename,
            pages=page_results,
            status="success",
        )
 
    except Exception as exc:
        print(f"[app] Error processing document: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(exc)}"
        )
 
    finally:
        # Always clean up the temp file even if an error occurred
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"[app] Cleaned up temp file: {tmp_path}")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)