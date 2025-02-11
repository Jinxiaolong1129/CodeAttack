import time
from utils import api_call, get_client


class TargetLLM:
    def __init__(self, model_name, max_tokens=512, seed=725, temperature=0.0):
        self.client = get_client(model_name)
        self.model_name = model_name
        self.max_retry = 3
        self.timeout = 200
        self.query_sleep = 20
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature

    def generate(self, query):
        for _ in range(self.max_retry):
            try:
                resp = api_call(client=self.client,
                                query=query,
                                model_name=self.model_name,
                                temperature=self.temperature)
                return resp
            except Exception as e:
                print("error", e)
                time.sleep(self.query_sleep)
        summ = "All retry attempts failed."
        return summ  # Or raise an exception if desired




from typing import Optional
import time
from vllm import LLM, SamplingParams

class TargetLLM_vllm:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        seed: int = 725,
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature
        self.max_retry = 3
        self.query_sleep = 20
        
        # Initialize vllm engine
        try:
            self.engine = LLM(
                model=model_name,
                trust_remote_code=True
            )
            print(f"Successfully loaded {model_name} with vllm")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vllm for {model_name}: {str(e)}")
            
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate(self, query: str) -> str:
        """Generate response using chat interface"""
        for attempt in range(self.max_retry):
            try:
                response = self.engine.chat(
                    messages=[{"role": "user", "content": query}],
                    sampling_params=self.sampling_params
                )
                output = response[0].outputs[0].text
                if output and len(output) > 0:
                    return output
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retry - 1:
                    time.sleep(self.query_sleep)
                continue
                
        return "All retry attempts failed."
