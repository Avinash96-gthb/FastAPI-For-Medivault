from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import pytesseract
from datetime import datetime
import os
from pathlib import Path
import PyPDF2
from transformers import pipeline
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfWriter, PdfReader

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
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{current_time}.pdf"
        size = Path(url).stat().st_size

        if size > 10000000:
            return {"error": "File is too large", "size_mb": size / 1000000}

        if url.lower().endswith('.pdf'):
            os.rename(url, filename)
            return {"message": "Already a PDF", "name": filename}

        a = cv2.imread(url)
        b = pytesseract.image_to_pdf_or_hocr(a, extension='pdf')

        with open(filename, 'wb') as f:
            f.write(b)

        return {"message": "File converted successfully", "name": filename}
    except Exception as e:
        return {"error": str(e)}

@app.put("/Summarization")
async def summarize(url: str):
    try:
        pdf_writer = PdfWriter()
        with open(url, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            summaries = []

            # Extract and summarize text from each page
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text.strip():
                    summary = summarizer(page_text, max_length=200, min_length=20, do_sample=False)
                    summaries.append({"page": page_num + 1, "summary": summary[0]['summary_text']})

                    # Add the original page
                    pdf_writer.add_page(pdf_reader.pages[page_num])

                    # Create a new PDF page with the summary
                    summary_pdf_path = f"summary_page_{page_num + 1}.pdf"
                    c = canvas.Canvas(summary_pdf_path, pagesize=letter)
                    text = summary[0]['summary_text']
                    c.drawString(72, 750, f"Summary for Page {page_num + 1}:")
                    c.drawString(72, 730, text)
                    c.save()

                    # Merge the summary page with the original PDF
                    with open(summary_pdf_path, "rb") as summary_pdf:
                        summary_reader = PdfReader(summary_pdf)
                        pdf_writer.add_page(summary_reader.pages[0])

                    # Clean up the temporary summary PDF file
                    os.remove(summary_pdf_path)

            # Save the output PDF with summaries
            output_filename = f"summarized_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
            with open(output_filename, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            return {"output_pdf": output_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
