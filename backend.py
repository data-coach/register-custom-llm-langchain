from h2ogpte import H2OGPTE
from config import storage
api_key = storage.get("h2ogpte_key")
url = storage.get("h2ogpte_url")

# Create Client
client = H2OGPTE(address=url,
                 api_key=api_key)

# Get all available LLMs
llm_names = [x["base_model"] for x in client.get_llms()]
# for i in range(len(llm_names)):
#     print(i+1,llm_names[i])
    
# Set model for next step
selected_model_name = "gpt-4o"

# Generate Response
prompt = str(input("ask something..."))
message = f"""[
    {{ "role": "system", "content": "You are a helpful assistant." }},
    {{"role":"user", "content":{prompt}}}
]"""

# Call LLM
response = client.answer_question(
    question=message,
    llm=selected_model_name,
    llm_args={
        "temperature":0,
        "max_new_tokens":1024,
        "response_format":"text"})

print(response)