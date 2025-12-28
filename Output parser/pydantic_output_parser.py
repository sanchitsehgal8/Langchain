from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='google/functiongemma-270m-it',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(..., description="The age of the person")
    city: str = Field(..., description="The city where the person lives")


parser=PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template="give me the name age and city of a fictional {place} person \n{format_instructions}",
    input_variables=["place" ],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


prompt = template.invoke({"place":"Indian"})
result = model.invoke(prompt)
final = parser.parse(result.content)
print(final)