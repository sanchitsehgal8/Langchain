from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template1 = PromptTemplate(
    template="give me the name age and city of a fictional person \mn {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# partial variabl are those which are filled and not provided at runtime
# json output parser is used to parse the output of the model into json format
#exceot string output all other parsers are given as foramt instructions to the prompt template
# way no 1 
result = llm.invoke(template1.format())

final = parser.parse(result)
# way no 2 chaining
# chain = template1 | model | parser
# final = chain.invoke({})    

print(final)

print(type(final))

# JSON cannon enforce any particular schema or structure on the output beyond ensuring that it is valid JSON.
# If you need to enforce a specific schema, consider using PydanticOutputParser or another parser that supports schema validation.