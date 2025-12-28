from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
os.environ['HF_HOME'] = 'D:/huggingface_cache'

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The classification of the feedback as either positive or negative")


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template="classify the following feedback text into positive or negative: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)


llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,        
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)    
parser = StrOutputParser()


classify_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write response to this positive feedback : {feedback}",
    input_variables=["feedback"]
) 
prompt3 = PromptTemplate(
    template="Write response to this negative feedback : {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x: "No valid sentiment found.")
     
)
chain = classify_chain | branch_chain
result = chain.invoke({"feedback": "This is the shittiest product I have ever used.I  will never buy this again!"})
print("Final Response:\n", result) 
chain.get_graph().print_ascii() 