# Initialize the model
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o"
)

# Batch prompt invocation
prompts = [
    "Tell me a joke.",
    "Who is APJ Abdul Kalam?",
    "What is Artificial Intelligence?."
]

# Get responses for multiple prompts
responses = model.batch(prompts)
for res in responses:
    print(res)