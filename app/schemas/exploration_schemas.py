from pydantic import BaseModel


class Exploration(BaseModel):
    name: str
    description: str
