from day1.llm_client import LLMClient
from day2.day2_utils import build_prompt

llm = LLMClient()
# Step 1｜设计多版本 Prompt
# task = "请总结以下文本核心观点：‘人工智能在金融风控的应用日益广泛。’"
#
# prompts = [
# build_prompt(role="分析师", task=task),
# build_prompt(role="严谨报告撰写者", task=task, output_format="JSON"),
# build_prompt(role="知识库助手", task=task, output_format="表格")
# ]
#
# for i, p in enumerate(prompts):
#     print(f"--- Prompt {i+1} ---")
#     resp = llm.chat(user_message=p)
#     print(resp)

# Step 2｜输出 JSON / 表格约束
# prompt_json = build_prompt(
# role="技术文档分析师",
# task=task,
# output_format='''{
# "summary": "一句话总结",
# "keywords": ["关键词1", "关键词2"]
# }'''
# )
#
# resp_json = llm.chat(user_message=prompt_json)
# print(resp_json)


# Step 3｜实现“不知道就说不知道”
prompt_safe = build_prompt(
role="技术助手",
task="请回答下列问题：如何保持平静？",
output_format="JSON"
) + "\\n如果无法确定，请回答：'不知道'"

resp_safe = llm.chat(user_message=prompt_safe)
print(resp_safe)


# Step 4｜构建 Prompt 模板库
prompt_registry = {
"summarize": build_prompt("分析师", "", "JSON"),
"qa_safe": build_prompt("技术助手", "", "JSON + 不知道"),
"table_output": build_prompt("知识库助手", "", "表格")
}
