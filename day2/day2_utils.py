from typing import Optional


def build_prompt(role: str, task: str, output_format: Optional[str] = None) -> str:
    """构建系统提示词

    Args:
        role: 角色描述，如 "分析师"、"知识库助手"
        task: 具体任务内容
        output_format: 输出格式要求，如 "JSON"、"表格"，默认不限定格式

    Returns:
        构建好的提示词字符串
    """
    prompt = f"你是一个{role}，请{task}"

    if output_format:
        prompt += f"\n请以{output_format}格式输出结果。"

    return prompt
