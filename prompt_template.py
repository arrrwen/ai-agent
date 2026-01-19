from string import Template
from typing import Dict, Any


class PromptTemplate:
    """Prompt 模板类，支持变量替换"""

    def __init__(self, template: str, **default_vars):
        """
        Args:
            template: 模板字符串，使用 {variable} 格式定义变量
            **default_vars: 默认变量值
        """
        self.template = template
        self.default_vars = default_vars

    def format(self, **kwargs) -> str:
        """格式化模板，用提供的变量替换占位符"""
        vars = {**self.default_vars, **kwargs}
        return self.template.format(**vars)

    def __call__(self, **kwargs) -> str:
        """支持直接调用"""
        return self.format(**kwargs)


# ============ 预定义模板 ============

# 通用助手
ASSISTANT_TEMPLATE = PromptTemplate(
    "你是一个{role}，请{task}。"
)

# 翻译模板
TRANSLATE_TEMPLATE = PromptTemplate(
    "请将以下内容翻译成{target_language}，只输出翻译结果：\n\n{text}"
)

# 摘要模板
SUMMARY_TEMPLATE = PromptTemplate(
    "请用{length}总结以下内容的要点：\n\n{text}",
    length="100字以内"
)

# 代码解释模板
CODE_EXPLAIN_TEMPLATE = PromptTemplate(
    "请用{language}解释以下代码的功能：\n\n```{code_lang}\n{code}\n```",
    language="中文",
    code_lang=""
)

# 问答模板（带上下文）
QA_TEMPLATE = PromptTemplate(
    """基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""
)

# 角色扮演模板
ROLEPLAY_TEMPLATE = PromptTemplate(
    """你现在扮演{character}。
背景设定：{background}
请以该角色的身份回应用户。保持角色一致性。"""
)


if __name__ == "__main__":
    # 测试示例
    from llm_client import LLMClient

    client = LLMClient()

    # 翻译示例
    prompt = TRANSLATE_TEMPLATE.format(
        target_language="英文",
        text="人工智能正在改变世界"
    )
    print("=== 翻译示例 ===")
    print(f"Prompt: {prompt}")
    print(f"Response: {client.chat(prompt)}\n")

    # 摘要示例
    prompt = SUMMARY_TEMPLATE.format(
        text="RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的技术，通过从外部知识库中检索相关信息来增强生成模型的输出准确性与上下文相关性。"
    )
    print("=== 摘要示例 ===")
    print(f"Prompt: {prompt}")
    print(f"Response: {client.chat(prompt)}\n")