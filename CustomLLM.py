'''
    自定义LLM
'''
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests


class CustomLLM(LLM):
    endpoint: str = "***/v1"  # 本地大模型的API地址
    model: str = "***" #大模型名称

    @property
    def _llm_type(self) -> str:
        return "***" #大模型名称

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
        ) -> str:
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(f"{self.endpoint}/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
            return "error"

        result = response.json()
        text = result["choices"][0]["message"]["content"]
        return text