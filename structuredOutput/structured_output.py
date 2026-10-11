import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated


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
# class Actor(BaseModel):
#     name: str
#     role: str

# class MovieDetails(BaseModel):
#     title:str
#     year: int
#     cast: list[Actor]
#     genres:list[str]
#     budget: float | None = Field(None, description="Budget in millions USD")

# model_with_structure = model.with_structured_output(MovieDetails)

# response = model_with_structure.invoke("Provide details about the movie Inception")
# print(response)

###### TypedDict ######
# class MovieDict(TypedDict):
#     title:Annotated[str, ..., "The title of the movie"]
#     year:Annotated[str, ..., "The year the movie was releasted"]
#     director:Annotated[str, ..., "The director of the movie"]
#     rating:Annotated[str, ..., "The movie's rating out of 10"]

# model_withtypeddict=model.with_structured_output(MovieDict)
# response = model_withtypeddict.invoke("Please provide the details of the movie Avengers")
# print(response)

# class Actor(TypedDict):
#     name: str
#     role: str
# class MovieDetails(TypedDict):
#     title:str
#     year: int
#     cast: list[Actor]
#     genres:list[str]
#     budget: float | None = Field(None, description="Budget in millions USD")
# model_with_structure = model.with_structured_output(MovieDetails)
# response = model_with_structure.invoke("Please provide the details of the movie Avengers")

# print(response)


###### Use of Agent ######

# class ContactInfo(BaseModel):
#     name: str = Field(description="The name of the person")
#     email: str = Field(description="The email address of the person")
#     phone: str = Field(description="The phone number of the person")

# agent = create_agent(
#     model="gpt-5",
#     response_format=ContactInfo
# )

# result = agent.invoke({
#     "messages": [{"role": "user", "content":"Extract content info from: John Doe, john@example.com, (555) 123-4567"}]
# })

# print(result["structured_response"])


###### Use of Dataclass ######
@dataclass
class ContactInfo:
    """Contact information for a person"""
    name: str 
    email: str  
    phone: str 

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{"role": "user", "content":"Extract content info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])