from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field 

load_dotenv()
model= ChatOpenAI(model='gpt-4')

# Pydantic Model with field descriptions 

class ReviewSummary(BaseModel):
    key_themes: list[str] = Field( description="The main themes discussed in the review.")
    summary: str = Field( description="A brief summary of the review.")
    sentiment: Literal["positive", "negative"] = Field( description="The overall sentiment of the review, e.g., positive, negative, neutral.")
    pros: Optional[list[str]] = Field(default=None, description="List of positive aspects mentioned in the review.")
    cons: Optional[list[str]] = Field(default=None,  description="List of negative aspects mentioned in the review.")
    name: Optional[str] = Field(default=None, description="Name of the reviewer.")


     
structured_model=  model.with_structured_output(ReviewSummary)
result = structured_model.invoke("The hardware specifications of MacBook Pro 2023 are very impressive.The ui has some issues though.")
print(type(result))
print(result)
print(result.summary)
print(result.sentiment)
print(result.pros)
print(result.cons)
print(result.name)     

# this is only for data reprentation but you cannon validate or can enforce conditions on data 
