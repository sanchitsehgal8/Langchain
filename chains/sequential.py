from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ['HF_HOME'] = 'D:/huggingface_cache'

prompt1 = PromptTemplate(
    template="generate a detailed report on : {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="summarize the following report in a 3 line summary: {report}",
    input_variables=["report"]
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

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "India"})
print("Final Summary:\n", result)
chain.get_graph().print_ascii()