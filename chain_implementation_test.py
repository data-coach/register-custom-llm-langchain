from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Initialize the model
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o"
)
# Define a prompt template for the chain
prompt_template = "What is the capital of {input_value}?"
prompt = PromptTemplate(input_variables=["input_value"], template=prompt_template)

# Create a processing chain
chain = prompt | model | StrOutputParser()

# Invoke the chain with an example input
print(chain.invoke("India"))