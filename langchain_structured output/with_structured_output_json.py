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
json_schema = {
    "title": "ReviewSummary",  
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief summary of the review."
        },     
        "sentiment": {
            "type": "string",
            "description": "The overall sentiment of the review, e.g., positive, negative, neutral."
        },      
        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of positive aspects mentioned in the review."
        },      
        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List of negative aspects mentioned in the review."
        }
    },
    "required": ["summary", "sentiment"]

}
structured_model=  model.with_structured_output(json_schema)
result = structured_model.invoke("The hardware specifications of MacBook Pro 2023 are very impressive.The ui has some issues though.")
print(type(result))
print(result)
print(result['summary'])
print(result['sentiment'])
print(result.get('pros'))
print(result.get('cons'))

# this is only for data reprentation but you cannon validate or can enforce conditions on data 
