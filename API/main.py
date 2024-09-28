from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import pytesseract
from datetime import datetime
import os
import shutil
from pathlib import Path
import PyPDF2
from transformers import pipeline
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the summarizer model
summarizer = pipeline('summarization', model='Falconsai/medical_summarization')

class SummaryResponse(BaseModel):
    page: int
    summary: str

class SummarizationResponse(BaseModel):
    summaries: List[SummaryResponse]

@app.post("/convert-to-pdf")
async def convert_to_pdf(file: UploadFile = File(...)):
    try:
        # Create a timestamped filename
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pdf_filename = f"{current_time}.pdf"
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save the uploaded file temporarily
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check if the file is already a PDF
        if file.filename.lower().endswith('.pdf'):
            pdf_path = file_location
        else:
            # Read the uploaded file using OpenCV
            image = cv2.imread(file_location)
            if image is None:
                raise HTTPException(status_code=400, detail="Unable to read the file as an image")

            # Convert the image to PDF using pytesseract
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

            # Save the converted PDF with the current date and time as the filename
            pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)

        # Return the PDF file as a response
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_filename)
    
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
