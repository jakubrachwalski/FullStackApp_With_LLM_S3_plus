from typing import List
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
import schemas
import crud
from database import SessionLocal
from uuid import uuid4

from schemas import QuestionRequest

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import boto3
import tempfile
import os
from config import Settings

from fastapi.responses import PlainTextResponse


router = APIRouter(prefix="/pdfs")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=schemas.PDFResponse, status_code=status.HTTP_201_CREATED)
def create_pdf(pdf: schemas.PDFRequest, db: Session = Depends(get_db)):
    return crud.create_pdf(db, pdf)

@router.post("/upload", response_model=schemas.PDFResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_name = f"{uuid4()}-{file.filename}"
    return crud.upload_pdf(db, file, file_name)

@router.get("", response_model=List[schemas.PDFResponse])
def get_pdfs(selected: bool = None, db: Session = Depends(get_db)):
    return crud.read_pdfs(db, selected)

@router.get("/{id}", response_model=schemas.PDFResponse)
def get_pdf_by_id(id: int, db: Session = Depends(get_db)):
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return pdf

@router.put("/{id}", response_model=schemas.PDFResponse)
def update_pdf(id: int, pdf: schemas.PDFRequest, db: Session = Depends(get_db)):
    updated_pdf = crud.update_pdf(db, id, pdf)
    if updated_pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return updated_pdf

@router.delete("/{id}", status_code=status.HTTP_200_OK)
def delete_pdf(id: int, db: Session = Depends(get_db)):
    if not crud.delete_pdf(db, id):
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"message": "PDF successfully deleted"}


# LANGCHAIN
langchain_llm = OpenAI(temperature=0)

summarize_template_string = """
        Provide a summary for the following text:
        {text}
"""

summarize_prompt = PromptTemplate(
    template=summarize_template_string,
    input_variables=['text'],
)

summarize_chain = summarize_prompt | langchain_llm

@router.post('/summarize-text')
async def summarize_text(text: str):
    summary = summarize_chain.invoke(text)
    return {'summary': summary}




@router.post("/qa-pdf/{id}", response_class=PlainTextResponse)
def qa_pdf_by_id(id: int, question_request: QuestionRequest, db: Session = Depends(get_db)):
    pdf = crud.read_pdf(db, id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    s3_url_prefix = f"https://{Settings().AWS_S3_BUCKET}.s3.amazonaws.com/"
    if not pdf.file.startswith(s3_url_prefix):
        raise HTTPException(status_code=400, detail="Invalid S3 URL format")
    s3_key = pdf.file[len(s3_url_prefix):]
    s3_client = Settings.get_s3_client()
    local_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            local_path = tmp.name
            s3_client.download_fileobj(Settings().AWS_S3_BUCKET, s3_key, tmp)
        loader = PyPDFLoader(local_path)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        document_chunks = text_splitter.split_documents(document)
        embeddings = OpenAIEmbeddings()
        stored_embeddings = FAISS.from_documents(document_chunks, embeddings)
        QA_chain = RetrievalQA.from_chain_type(
            llm=langchain_llm,
            chain_type="stuff",
            retriever=stored_embeddings.as_retriever()
        )
        question = question_request.question
        answer = QA_chain.run(question)
        # Ensure the answer is a string
        if isinstance(answer, dict):
            answer = answer.get('result') or answer.get('answer') or str(answer)
        return answer
    except Exception as e:
        print(f"Error in qa_pdf_by_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)