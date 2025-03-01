# Initialize the model
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o"
)

# Simple test with a direct query
response = model.invoke("What is the capital of France?")
print(response)
