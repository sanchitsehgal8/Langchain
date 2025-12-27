# pydantic is a data validation and data parsing library for python.
#It ensures data you work withis correct,structured and adheres to defined schema.
from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name: str
    age: Optional[int] = None
    
    cpga: float=Field(gt=0.0,lt=4.0,description="Cumulative Grade Point Average")
    
new_student= {'name':'Sanchit','age':21, 'email':'sanchit@example.com', 'cpga':3.8}

student= Student(**new_student)
print(student)

 #in this if you put name as integer it will raise validation error