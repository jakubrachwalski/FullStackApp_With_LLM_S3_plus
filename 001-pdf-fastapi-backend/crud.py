from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
import models, schemas
from config import Settings
from botocore.exceptions import NoCredentialsError, BotoCoreError


def create_pdf(db: Session, pdf: schemas.PDFRequest):
    db_pdf = models.PDF(name=pdf.name, selected=pdf.selected, file=pdf.file)
    db.add(db_pdf)
    db.commit()
    db.refresh(db_pdf)
    return db_pdf

def read_pdfs(db: Session, selected: bool = None):
    if selected is None:
        return db.query(models.PDF).all()
    else:
        return db.query(models.PDF).filter(models.PDF.selected == selected).all()

def read_pdf(db: Session, id: int):
    return db.query(models.PDF).filter(models.PDF.id == id).first()

def upload_pdf(db: Session, file: UploadFile, file_name: str):
    s3_client = Settings.get_s3_client()
    BUCKET_NAME = Settings().AWS_S3_BUCKET
    
    try:
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            file_name,
            ExtraArgs={'ACL': 'public-read'}  # <-- THIS LINE MAKES IT PUBLIC
        )
        file_url = f'https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}'
        
        db_pdf = models.PDF(name=file.filename, selected=False, file=file_url)
        db.add(db_pdf)
        db.commit()
        db.refresh(db_pdf)
        return db_pdf
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Error in AWS credentials")

def delete_pdf(db: Session, id: int):
    db_pdf = db.query(models.PDF).filter(models.PDF.id == id).first()
    if db_pdf is None:
        return None

    # Delete from S3
    from config import Settings
    import re
    s3_client = Settings.get_s3_client()
    bucket_name = Settings().AWS_S3_BUCKET

    # Extract S3 key from the URL
    s3_url_prefix = f"https://{bucket_name}.s3.amazonaws.com/"
    if db_pdf.file.startswith(s3_url_prefix):
        s3_key = db_pdf.file[len(s3_url_prefix):]
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        except Exception as e:
            print(f"Error deleting file from S3: {e}")
            # Optionally handle/log or ignore if file doesn't exist

    db.delete(db_pdf)
    db.commit()
    return True

def upload_pdf(db: Session, file: UploadFile, file_name: str):
    s3_client = Settings.get_s3_client()
    BUCKET_NAME = Settings().AWS_S3_BUCKET
    
    try:
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            file_name
        )
        file_url = f'https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}'
        
        db_pdf = models.PDF(name=file.filename, selected=False, file=file_url)
        db.add(db_pdf)
        db.commit()
        db.refresh(db_pdf)
        return db_pdf
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Error in AWS credentials")


# def upload_pdf(db: Session, file: UploadFile, file_name: str):
#     s3_client = Settings.get_s3_client()
#     BUCKET_NAME = Settings().AWS_S3_BUCKET

#     try:
#         s3_client.upload_fileobj(
#             file.file,
#             BUCKET_NAME,
#             file_name,
#             ExtraArgs={'ACL': 'public-read'}
#         )
#         file_url = f'https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}'
        
#         db_pdf = models.PDF(name=file.filename, selected=False, file=file_url)
#         db.add(db_pdf)
#         db.commit()
#         db.refresh(db_pdf)
#         return db_pdf
#     except NoCredentialsError:
#         raise HTTPException(status_code=500, detail="Error in AWS credentials")
#     except BotoCoreError as e:
#         raise HTTPException(status_code=500, detail=str(e))
