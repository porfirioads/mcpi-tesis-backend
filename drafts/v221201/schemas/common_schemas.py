from pydantic import BaseModel


class FileUpload(BaseModel):
    file_path: str
