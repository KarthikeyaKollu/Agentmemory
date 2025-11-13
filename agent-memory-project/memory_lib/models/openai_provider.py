import os
import json
from openai import OpenAI
from typing import List, Type, Optional
from ..interfaces import BaseModelProvider
from ..schemas import Message, BaseModel, MemoryUpdatePlan

class OpenAIProvider(BaseModelProvider):
    """A real implementation of the AI provider using OpenAI."""
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        print(f"[OpenAIProvider] Initialized with model: {self.model}")

    def get_structured_completion(self, messages: List[Message], output_model: Type[BaseModel]) -> BaseModel:
        """
        Gets a real, structured JSON completion from the OpenAI API.
        This is not a mock.
        """
        print("[OpenAIProvider] Getting structured completion from API...")
        
        api_messages = [msg.model_dump() for msg in messages]

        # Use the JSON mode with schema support (for compatible models)
        tool_choice = {
            "type": "function",
            "function": {"name": "memory_update_plan"}
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "memory_update_plan",
                    "description": "The plan of actions to take to update the user's memory.",
                    "parameters": output_model.model_json_schema()
                }
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice
            )
            
            # The response will be a tool call, not content
            tool_call = response.choices[0].message.tool_calls[0]
            response_args = tool_call.function.arguments
            
            if not response_args:
                raise ValueError("Received empty tool call arguments from API.")
                
            print("[OpenAIProvider] Received and parsing tool call JSON response.")
            
            # Parse the raw JSON string from the tool call
            parsed_data = json.loads(response_args)
            return output_model.model_validate(parsed_data)

        except Exception as e:
            print(f"[OpenAIProvider] CRITICAL ERROR: {e}")
            print(f"[OpenAIProvider] Returning an empty plan to prevent errors.")
            # On failure, return an empty plan to avoid crashing the manager
            return MemoryUpdatePlan(plan=[])