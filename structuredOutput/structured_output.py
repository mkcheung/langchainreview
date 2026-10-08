import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title:str=Field(description="The title of the movie")
    year:int=Field(description="This is the year the movie was released")
    director:str=Field(description="The director of the movie")
    ratings:float=Field(description="The movie's ratings out of 10")

model = init_chat_model("google_genai:gemini-3.5-flash-lite")

# model_with_structure=model.with_structured_output(Movie)

# response = model_with_structure.invoke("Provide details about the movie Inception")

# print(response)

###### Demonstrate nested structure ######
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title:str
    year: int
    cast: list[Actor]
    genres:list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)
