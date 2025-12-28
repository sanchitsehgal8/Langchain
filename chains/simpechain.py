from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ['HF_HOME'] = 'D:/huggingface_cache'

prompt = PromptTemplate(
    template="generate 5 facts about : {topic}",
    input_variables=["topic"]
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
chain = prompt | model | parser
result = chain.invoke({"topic": "Python programming language"})
print("Final Facts:\n", result)

chain.get_graph().print_ascii()