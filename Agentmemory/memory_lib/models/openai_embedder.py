import os
from openai import OpenAI
from typing import List, Optional
from ..interfaces import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """Creates embeddings using OpenAI."""
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        print(f"[OpenAIEmbedder] Initialized with model: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string of text."""
        text = text.replace("\n", " ") # Per OpenAI recommendation
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding