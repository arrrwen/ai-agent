from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class LLMClient:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str = None,
        base_url: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        temperature: float = None,
    ) -> str:
        """发送消息并获取回复"""
        messages = []
        prompt = system_prompt or self.system_prompt
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": user_message})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content

    def chat_with_history(
        self,
        messages: list,
        temperature: float = None,
    ) -> str:
        """支持多轮对话，传入完整消息历史"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    client = LLMClient(system_prompt="你是一个严谨的技术助手")
    response = client.chat("用一句话解释什么是 RAG")
    print(response)