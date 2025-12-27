from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv
import os
load_dotenv()
model= ChatOpenAI(model='gpt-4')

# simple TypedDict without annotations
# class ReviewSummary(TypedDict):
#     summary: str
#     sentiment: str

class ReviewSummary(TypedDict):
    key_themes: Annotated[list[str], "The main themes discussed in the review."]
    summary: Annotated[str, "A brief summary of the review."]
    sentiment: Annotated[Literal["positive", "negative"], "The overall sentiment of the review, e.g., positive, negative, neutral."]
    pros: Annotated[Optional[list[str]], "List of positive aspects mentioned in the review."]   
    cons: Annotated[Optional[list[str]], "List of negative aspects mentioned in the review."]

structured_model=  model.with_structured_output(ReviewSummary)
result = structured_model.invoke("The hardware specifications of MacBook Pro 2023 are very impressive.The ui has some issues though.")
print(type(result))
print(result)
print(result['summary'])
print(result['sentiment'])
print(result.get('pros'))
print(result.get('cons'))

# this is only for data reprentation but you cannon validate or can enforce conditions on data 
