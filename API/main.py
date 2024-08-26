from fastapi import FastAPI, HTTPException
import cv2
import pytesseract
from datetime import datetime
import os
from pathlib import Path
import requests
FileName=[]
app = FastAPI()

SUPABASE_URL = 'https://fadgcptleqcskuvdxuri.supabase.co'
SUPABASE_API = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhZGdjcHRsZXFjc2t1dmR4dXJpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjAwNzAwMjQsImV4cCI6MjAzNTY0NjAyNH0.l8dg0upUoIHDkAOOMvQ5A_LDe8b0zkkTzi1buQG8Hfg'

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

