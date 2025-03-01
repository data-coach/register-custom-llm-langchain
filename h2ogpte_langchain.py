from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from h2ogpte import H2OGPTE
from config import storage

class CustomH2OGPTE(LLM):
    """A custom LLM wrapper for the H2OGPTE model to be used within LangChain."""
    
    api_key: str = storage.get("h2ogpte_key")
    url: str = storage.get("h2ogpte_url")
    client: H2OGPTE = H2OGPTE(address=url, api_key=api_key)
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.07
    max_new_tokens: int = 1024
    min_max_new_tokens: int = 512
    response_format: str = "text"

    def get_llms(self) -> List[str]:
        """Returns a list of available model names from the H2OGPTE client."""
        return [x["base_model"] for x in self.client.get_llms()]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using the H2OGPTE client."""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        message = f"""{prompt}"""
        response = self.client.answer_question(
            question=message,
            llm=self.model_name,
            llm_args={
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_new_tokens,
                "min_max_new_tokens": self.min_max_new_tokens,
                "response_format": self.response_format,
            }
        )
        return response.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "response_format": self.response_format,
        }

    @property
    def _llm_type(self) -> str:
        return "h2ogpte"
