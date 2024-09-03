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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

app = FastAPI()

# Initialize the summarizer model
summarizer = pipeline('summarization', model='Falconsai/medical_summarization')

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
        file_path = Path(url)

        # Check if the file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        size = file_path.stat().st_size
        
        if size > 10000000:  # 10 MB limit
            return {"error": "File is too large", "size_mb": size / 1000000}

        if url.lower().endswith('.pdf'):
            return {"message": "Already a PDF", "name": filename}

        # Read image and convert to PDF
        a = cv2.imread(url)
        if a is None:
            raise HTTPException(status_code=400, detail="Unable to read image file")
        
        b = pytesseract.image_to_pdf_or_hocr(a, extension='pdf')

        with open(filename, 'wb') as f:
            f.write(b)

        return {"message": "File converted successfully", "name": filename}
    except Exception as e:
        return {"error": str(e)}

def wrap_text(text, max_width, font_size, canvas):
    """
    Wraps text for PDF canvas according to maximum width.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        canvas.setFont("Helvetica", font_size)
        if canvas.stringWidth(test_line) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

@app.put("/Summarization")
async def summarize(url: str):
    try:
        pdf_writer = PdfWriter()
        summaries = []
        
        # Check if the file exists
        file_path = Path(url)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(url, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            
            num_pages = len(pdf_reader.pages)
            if num_pages == 0:
                raise HTTPException(status_code=400, detail="PDF has no pages")
            
            # Extract and summarize text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                
                if page_text.strip():
                    # Adjust summarization parameters if needed
                    summary = summarizer(page_text, max_length=500, min_length=100, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    # Print the summary for debugging
                    print(f"Summary for Page {page_num + 1}: {summary_text}")

                    summaries.append({"page": page_num + 1, "summary": summary_text})

                    # Add the original page
                    pdf_writer.add_page(page)

                    # Create a new PDF page with the summary
                    summary_pdf_path = f"summary_page_{page_num + 1}.pdf"
                    c = canvas.Canvas(summary_pdf_path, pagesize=letter)
                    text_object = c.beginText(72, 750)
                    text_object.setFont("Helvetica", 10)

                    # Wrap the summary text
                    wrapped_text = wrap_text(summary_text, max_width=480, font_size=10, canvas=c)

                    y_position = 750
                    for line in wrapped_text:
                        if y_position < 50:  # Start a new page if necessary
                            c.showPage()
                            y_position = 750
                            text_object = c.beginText(72, y_position)
                            text_object.setFont("Helvetica", 10)
                        text_object.textLine(line)
                        y_position -= 12
                    
                    c.drawText(text_object)
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
