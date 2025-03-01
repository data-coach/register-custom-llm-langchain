from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o"
)
# Create a chat prompt template
chat = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Assistant with a sense of humor"),
    ("human", "Hi, how are you?"),
    ("ai", "I am good. How can I help you?"),
    ("human", "{input}")
])

# Format the prompt and generate a response
chat1 = chat.format_messages(input="What is the capital of South Africa?")
print(model.invoke(chat1))
