# Initialize the model
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o"
)

# Fetch and display available models
model_names = model.get_llms()
print("Available models:", model_names)
