# Registering a Custom LLM with Langchain

## Introduction
This repository provides a step-by-step guide on how to integrate a custom Large Language Model (LLM) with Langchain using a custom wrapper class. The example implementation uses the H2OGPTE model, demonstrating how to create, configure, and test a custom LLM.

## Features
- Custom LLM wrapper for H2OGPTE
- Easy integration with Langchain tools and chains
- Support for batch prompts and advanced LLM parameters

## Prerequisites
- Python 3.8+
- API Key and URL for H2OGPTE model

## Installation
Clone the repository and install the dependencies:
```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Setup
Ensure your API key and URL are stored securely in `config.storage`. The `storage.get()` method is used in the code to retrieve these values.

## Usage
### 1. Initialize the Model
```python
from config import storage
from h2ogpte_langchain import CustomH2OGPTE

model = CustomH2OGPTE(
    api_key=storage.get("h2ogpte_key"),
    url=storage.get("h2ogpte_url"),
    model_name="gpt-4o",
    temperature=0.7,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.05,
    max_new_tokens=512,
    min_max_new_tokens=256,
    response_format="text"
)
```

### 2. Simple Response
```python
response = model.invoke("What is the capital of France?")
print(response)
```

### 3. Batch Prompt Invocation
```python
prompts = ["Tell me a joke.", "What is the weather today?", "Explain quantum computing."]
responses = model.batch(prompts)
for response in responses:
    print(response)
```

### 4. Using with Langchain Tools & Chains
```python
from langchain_core.prompts import ChatPromptTemplate

chat = ChatPromptTemplate.from_messages([
   ("system","You are a helpful AI Assistant with a sense of humor"),
   ("human","Hi how are you?"),
   ("ai","I am good. How can I help you?"),
   ("human","{input}")
])

chat1 = chat.format_messages(input="What is the capital of South Africa?")
print(model.invoke(chat1))
```

### 5. Implementing a Chain
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt_template = "What is the capital of {input_value}?"
prompt = PromptTemplate(
    input_variables=["input_value"], template=prompt_template
)

chain = prompt | model | StrOutputParser()
print(chain.invoke("South Africa"))
```

### 6. Get Available Models
```python
model_names = model.get_llms()
print("Available models:", model_names)
```

## Contributing
Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Resources
- [Langchain Documentation](https://python.langchain.com/)
- [H2OGPTE Documentation](https://h2oai.github.io/h2ogpte/index.html#)

