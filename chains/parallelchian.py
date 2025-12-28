from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os   
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

os.environ['HF_HOME'] = 'D:/huggingface_cache'

prompt1 = PromptTemplate(
    template="generate short and simple notes from the given text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template= "generate 5 short questions from the following notes: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template= "merge the provided notes and questions into a single comprehensive study guide: \nNotes: {notes} \n Questions: {quiz}",
    input_variables=["notes","quiz"]
)


llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,        
        max_new_tokens=100
    )
)
model= ChatHuggingFace(llm=llm)   
 
parser = StrOutputParser() 


parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)

merge_chain =  prompt3 | model | parser
chain = parallel_chain | merge_chain

result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models. It enables developers to build applications that can understand and generate human-like text by leveraging large language models. LangChain provides tools and abstractions to simplify the integration of language models into various applications, making it easier to create conversational agents, chatbots, and other AI-driven solutions."})
print("Final Study Guide:\n", result)   

chain.get_graph().print_ascii()