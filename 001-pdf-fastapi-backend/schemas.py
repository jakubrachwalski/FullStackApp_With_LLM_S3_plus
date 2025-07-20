from pydantic import BaseModel

class PDFRequest(BaseModel):
    name: str
    selected: bool
    file: str

class PDFResponse(BaseModel):
    id: int
    name: str
    selected: bool
    file: str

    class Config:
        from_attributes = True
