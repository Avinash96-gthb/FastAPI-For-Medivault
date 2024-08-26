from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import pytesseract
from datetime import datetime
import os
from pathlib import Path
import PyPDF2
from transformers import pipeline
from typing import List, Dict

app = FastAPI()

# Initialize the summarizer model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

class SummaryResponse(BaseModel):
    page: int
    summary: str

class SummarizationResponse(BaseModel):
    summaries: List[SummaryResponse]

@app.put("/Convert_to_pdf")
async def convert_to_pdf(url: str):
    try:
        # Define the current time and filename
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{current_time}.pdf"
        size = Path(url).stat().st_size

        # Check file size
        if size > 10000000:
            return {"error": "File is too large", "size_mb": size / 1000000}

        # Check if the file is already a PDF
        if url.lower().endswith('.pdf'):
            os.rename(url, filename)
            return {"message": "Already a PDF", "name": filename}

        # Convert image to PDF
        a = cv2.imread(url)
        b = pytesseract.image_to_pdf_or_hocr(a, extension='pdf')

        with open(filename, 'wb') as f:
            f.write(b)

        return {"message": "File converted successfully", "name": filename}
    except Exception as e:
        return {"error": str(e)}

@app.put("/Summarization", response_model=SummarizationResponse)
async def summarize(url: str) -> SummarizationResponse:
    try:
        # Open and read the PDF file
        with open(url, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            summaries = []

            # Extract and summarize text from each page
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text.strip():  # Only summarize if there's text
                    summary = summarizer(page_text, max_length=200, min_length=20, do_sample=False)
                    summaries.append({"page": page_num + 1, "summary": summary[0]['summary_text']})

            return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
