"""
评论分析结果提炼工具
读取Excel中的用户痛点、用户需求、总结三列，用LLM提炼成短句
"""
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from day1.llm_client import LLMClient


def refine_text(llm: LLMClient, text: str, field_type: str) -> str:
    """使用LLM将文本提炼成短句"""
    if pd.isna(text) or not str(text).strip():
        return ""

    prompt = f"""请将以下{field_type}内容提炼成简洁的短句，每个要点用一句话概括，保留核心信息。

原文：
{text}

要求：
1. 每个要点提炼成一句简短的话（10-20字）
2. 保留关键信息，去除冗余
3. 用分号(；)分隔各个要点
4. 直接输出提炼结果，不要有其他说明"""

    system_prompt = "你是一个文本提炼专家，擅长将长文本精简为简洁有力的短句。"

    try:
        response = llm.chat(prompt, system_prompt=system_prompt, temperature=0.3)
        return response.strip()
    except Exception as e:
        print(f"  提炼失败: {e}")
        return text


def main():
    print("=" * 60)
    print("评论分析结果提炼工具")
    print("=" * 60)

    # 文件路径
    input_path = os.path.join(os.path.dirname(__file__), '评论分析结果.xlsx')
    output_path = os.path.join(os.path.dirname(__file__), '评论分析结果_提炼版.xlsx')

    # 读取Excel
    print(f"\n[1/3] 读取文件: {input_path}")
    df = pd.read_excel(input_path)
    print(f"共 {len(df)} 条记录")

    # 初始化LLM
    print("[2/3] 初始化LLM客户端...")
    llm = LLMClient(model="qwen-plus", temperature=0.3)

    # 提炼三列内容
    print("[3/3] 开始提炼...")

    for i in range(len(df)):
        print(f"  处理第 {i+1}/{len(df)} 条...")

        # 提炼用户痛点
        if pd.notna(df.loc[i, '用户痛点']):
            df.loc[i, '用户痛点'] = refine_text(llm, df.loc[i, '用户痛点'], '用户痛点')

        # 提炼用户需求
        if pd.notna(df.loc[i, '用户需求']):
            df.loc[i, '用户需求'] = refine_text(llm, df.loc[i, '用户需求'], '用户需求')

        # 提炼总结
        if pd.notna(df.loc[i, '总结']):
            df.loc[i, '总结'] = refine_text(llm, df.loc[i, '总结'], '总结')

    # 保存结果
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='评论分析结果')

        worksheet = writer.sheets['评论分析结果']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 12
        worksheet.column_dimensions['C'].width = 50
        worksheet.column_dimensions['D'].width = 50
        worksheet.column_dimensions['E'].width = 60

    print(f"\n提炼完成！结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
