from fastapi import FastAPI, HTTPException
import cv2
import pytesseract
from datetime import datetime
import os
from pathlib import Path
import requests
FileName=[]
app = FastAPI()


@app.put("/")
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
        
        with open(filename, 'w+b') as f:
            f.write(b)
        FileName.append(filename)
        return filename
    except Exception as e:
        return {"error": str(e)}


print(FileName)

