"""
RAG (Retrieval-Augmented Generation) Pipeline Demo
使用 FAISS 进行向量检索，结合 LLM 生成回答
"""

import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def get_client():
    """创建 OpenAI 兼容的客户端（使用阿里云 DashScope）"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    return client


def split_text(text, chunk_size=300, overlap=50):
    """
    将长文本分割成多个小块
    Args:
        text: 原始文本
        chunk_size: 每个块的字符数
        overlap: 块之间的重叠字符数
    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_texts(texts, client):
    """
    为文本列表生成向量嵌入
    Args:
        texts: 文本列表
        client: OpenAI 客户端
    Returns:
        嵌入向量列表
    """
    embeddings = []
    for t in texts:
        emb = client.embeddings.create(model="text-embedding-v3", input=t)
        embeddings.append(emb.data[0].embedding)
    return embeddings


def build_index(embeddings):
    """
    构建 FAISS 向量索引
    Args:
        embeddings: 嵌入向量列表
    Returns:
        FAISS 索引
    """
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index


def search(query, index, chunks, client, top_k=3):
    """
    检索与查询最相关的文本块
    Args:
        query: 查询文本
        index: FAISS 索引
        chunks: 原始文本块列表
        client: OpenAI 客户端
        top_k: 返回的最相关文本块数量
    Returns:
        最相关的文本块列表
    """
    query_embedding = embed_texts([query], client)[0]
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "distance": float(distances[0][i])
            })
    return results


def generate_answer(query, context_chunks, client):
    """
    使用 LLM 基于检索到的上下文生成回答
    Args:
        query: 用户问题
        context_chunks: 检索到的相关文本块
        client: OpenAI 客户端
    Returns:
        生成的回答
    """
    context = "\n---\n".join([c["chunk"] for c in context_chunks])

    prompt = f"""基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请说明无法从提供的资料中找到答案。

参考资料：
{context}

用户问题：{query}

请给出准确、简洁的回答："""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，根据提供的参考资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def rag_query(query, index, chunks, client, top_k=3):
    """
    完整的 RAG 查询流程
    Args:
        query: 用户问题
        index: FAISS 索引
        chunks: 文本块列表
        client: OpenAI 客户端
        top_k: 检索的文本块数量
    Returns:
        包含答案和检索结果的字典
    """
    # 1. 检索相关文本块
    search_results = search(query, index, chunks, client, top_k)

    # 2. 生成回答
    answer = generate_answer(query, search_results, client)

    return {
        "query": query,
        "answer": answer,
        "sources": search_results
    }


# ==================== Demo ====================

# 示例知识库文档
SAMPLE_DOCUMENT = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。
AI的历史可以追溯到1956年的达特茅斯会议，这被认为是人工智能作为学科诞生的时刻。

机器学习是人工智能的一个重要子领域，它使计算机能够从数据中学习，而无需进行明确的编程。
深度学习是机器学习的一个分支，使用多层神经网络来处理复杂的模式识别任务。

自然语言处理（NLP）是AI的另一个重要领域，专注于让计算机理解和生成人类语言。
大型语言模型（LLM）如GPT和Claude是NLP领域的重大突破，能够进行对话、写作和推理等任务。

RAG（检索增强生成）是一种结合检索和生成的技术。它首先从知识库中检索相关信息，然后将这些信息作为上下文提供给语言模型，以生成更准确和有依据的回答。
RAG的优势在于可以利用外部知识库，减少模型幻觉，并且可以随时更新知识而无需重新训练模型。

向量数据库是存储和检索向量嵌入的专用数据库。常见的向量数据库包括FAISS、Pinecone、Milvus和Chroma等。
FAISS是Facebook开发的开源库，用于高效的相似性搜索和稠密向量聚类。

Python比Java更有优势

陈信宏是这个世界上最帅的男人
"""


def main(demo_mode=False):
    """
    主函数 - 运行 RAG Demo
    Args:
        demo_mode: 如果为 True，运行预设问题演示；否则进入交互模式
    """
    print("=" * 50)
    print("RAG Pipeline Demo")
    print("=" * 50)

    # 初始化客户端
    print("\n[1] 初始化客户端...")
    client = get_client()

    # 分割文本
    print("[2] 分割文档...")
    chunks = split_text(SAMPLE_DOCUMENT, chunk_size=200, overlap=30)
    print(f"    文档被分割为 {len(chunks)} 个文本块")

    # 生成嵌入向量
    print("[3] 生成向量嵌入...")
    embeddings = embed_texts(chunks, client)
    print(f"    生成了 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")

    # 构建索引
    print("[4] 构建 FAISS 索引...")
    index = build_index(embeddings)
    print(f"    索引中包含 {index.ntotal} 个向量")

    # Demo 模式：运行预设问题
    if demo_mode:
        demo_questions = [
            "什么是 RAG？",
            "FAISS 是什么？",
            "深度学习和机器学习有什么关系？"
        ]
        print("\n" + "=" * 50)
        print("运行 Demo 模式，测试预设问题")
        print("=" * 50)

        for question in demo_questions:
            print(f"\n问题: {question}")
            print("-" * 40)
            result = rag_query(question, index, chunks, client)
            print(f"回答: {result['answer']}")
            print("\n参考来源:")
            for i, source in enumerate(result["sources"], 1):
                print(f"  [{i}] (距离: {source['distance']:.4f})")
                print(f"      {source['chunk'][:80]}...")
            print()
        return

    # 交互式查询
    print("\n" + "=" * 50)
    print("知识库已就绪！输入问题进行查询（输入 'quit' 退出）")
    print("=" * 50)

    while True:
        try:
            query = input("\n请输入问题: ").strip()
        except EOFError:
            print("\n检测到非交互环境，切换到 Demo 模式...")
            main(demo_mode=True)
            return

        if query.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break
        if not query:
            continue

        print("\n正在检索和生成回答...")
        result = rag_query(query, index, chunks, client)

        print("\n" + "-" * 40)
        print("回答：")
        print(result["answer"])
        print("\n参考来源：")
        for i, source in enumerate(result["sources"], 1):
            print(f"  [{i}] (距离: {source['distance']:.4f})")
            print(f"      {source['chunk'][:100]}...")


if __name__ == "__main__":
    import sys
    # 如果传入 --demo 参数，直接运行演示模式
    demo_mode = "--demo" in sys.argv
    main(demo_mode=demo_mode)
