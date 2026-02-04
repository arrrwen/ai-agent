# Question → Rewrite → Search → Rerank → Answer
# 为什么需要rewrite?

"""
用户的问题：

“这个接口为什么慢？”

文档里的表述：

“请求延迟主要由 IO 阻塞导致”

原始问题 embedding 很可能检索不到。
"""


from day1.llm_client import LLMClient
llm = LLMClient()
question = 'INTJ女python开发如何规划职业道路？'
# rewrite_prompt = f"""
# 你是搜索助手，请将用户问题改写成更适合检索文档的查询语句。
# 保留技术关键词，不要回答问题。
#
# 用户问题：{question}
# """

rewritten_query = llm.chat(user_message=question)

print(rewritten_query)
