"""
商品评论痛点需求分析工具
从数据库读取商品评论，使用LLM分析用户痛点和需求，结果导出到Excel
"""
import pyodbc
import pandas as pd
from typing import List, Dict
import sys
import os

# 添加项目根目录到路径，以便导入 LLMClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from day1.llm_client import LLMClient

# 数据库配置
SQL_SERVER_CONFIG = {
    'SERVER': '121.46.231.162,51433',
    'DATABASE': 'Compass_BP',
    'UID': 'compassBase',
    'PWD': 'hSG55#@$8D2AHJF;',
    'USE_WINDOWS_AUTH': False,
}


def get_db_connection():
    """创建数据库连接"""
    if SQL_SERVER_CONFIG['USE_WINDOWS_AUTH']:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER_CONFIG['SERVER']};"
            f"DATABASE={SQL_SERVER_CONFIG['DATABASE']};"
            f"Trusted_Connection=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER_CONFIG['SERVER']};"
            f"DATABASE={SQL_SERVER_CONFIG['DATABASE']};"
            f"UID={SQL_SERVER_CONFIG['UID']};"
            f"PWD={SQL_SERVER_CONFIG['PWD']};"
        )
    return pyodbc.connect(conn_str)


def fetch_random_products(conn, limit: int = 50) -> List[str]:
    """获取随机的商品ID列表"""
    query = f"""
        SELECT TOP {limit} product_id
        FROM (
            SELECT DISTINCT product_id
            FROM dsa_jd_comments_detail
        ) t
        ORDER BY NEWID()
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def fetch_comments_by_product(conn, product_id: str) -> pd.DataFrame:
    """获取指定商品的所有评论"""
    query = """
        SELECT * FROM dsa_jd_comments_detail
        WHERE product_id = ?
    """
    return pd.read_sql(query, conn, params=[product_id])


def analyze_product_comments(llm: LLMClient, product_id: str, comments_df: pd.DataFrame) -> Dict:
    """使用LLM分析单个商品的评论，提取痛点和需求"""
    # 提取评论内容（假设评论字段名为 content 或 comment，需根据实际字段调整）
    comment_columns = ['content', 'comment', 'comment_content', 'review', 'comment_text']
    comment_col = None
    for col in comment_columns:
        if col in comments_df.columns:
            comment_col = col
            break

    if comment_col is None:
        # 尝试找包含 comment 的列
        for col in comments_df.columns:
            if 'comment' in col.lower() or 'content' in col.lower():
                comment_col = col
                break

    if comment_col is None:
        return {
            'product_id': product_id,
            'comment_count': len(comments_df),
            'pain_points': '无法识别评论字段',
            'user_needs': '无法识别评论字段',
            'summary': f'可用字段: {list(comments_df.columns)}'
        }

    # 获取评论文本，限制数量避免token过长
    comments = comments_df[comment_col].dropna().tolist()
    if len(comments) > 100:
        comments = comments[:100]  # 限制最多100条评论

    comments_text = "\n".join([f"- {c}" for c in comments if c and str(c).strip()])

    if not comments_text.strip():
        return {
            'product_id': product_id,
            'comment_count': len(comments_df),
            'pain_points': '无有效评论内容',
            'user_needs': '无有效评论内容',
            'summary': '评论内容为空'
        }

    prompt = f"""请分析以下商品评论，提取用户的痛点和需求。

商品ID: {product_id}
评论数量: {len(comments_df)}

评论内容:
{comments_text}

请按以下格式输出分析结果:
1. 用户痛点（列出主要痛点，每个痛点一行）
2. 用户需求（列出用户期望的改进或功能，每个需求一行）
3. 总结（一段话概括该商品的用户反馈情况）

请直接输出分析结果，不要有其他内容。"""

    system_prompt = """你是一个专业的电商评论分析师，擅长从用户评论中提取有价值的信息。
你需要客观、准确地分析评论内容，识别用户的真实痛点和需求。
输出要简洁明了，重点突出。"""

    try:
        response = llm.chat(prompt, system_prompt=system_prompt, temperature=0.3)

        # 解析响应
        pain_points = ""
        user_needs = ""
        summary = ""

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if '痛点' in line:
                current_section = 'pain_points'
                continue
            elif '需求' in line:
                current_section = 'user_needs'
                continue
            elif '总结' in line:
                current_section = 'summary'
                continue

            if line and current_section == 'pain_points':
                pain_points += line + "\n"
            elif line and current_section == 'user_needs':
                user_needs += line + "\n"
            elif line and current_section == 'summary':
                summary += line + " "

        return {
            'product_id': product_id,
            'comment_count': len(comments_df),
            'pain_points': pain_points.strip() or response,
            'user_needs': user_needs.strip() or '',
            'summary': summary.strip() or ''
        }

    except Exception as e:
        return {
            'product_id': product_id,
            'comment_count': len(comments_df),
            'pain_points': f'分析失败: {str(e)}',
            'user_needs': '',
            'summary': ''
        }


def save_to_excel(results: List[Dict], output_path: str):
    """将分析结果保存到Excel"""
    df = pd.DataFrame(results)
    df.columns = ['商品ID', '评论数量', '用户痛点', '用户需求', '总结']

    # 使用 openpyxl 引擎保存，支持中文
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='评论分析结果')

        # 调整列宽
        worksheet = writer.sheets['评论分析结果']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 12
        worksheet.column_dimensions['C'].width = 50
        worksheet.column_dimensions['D'].width = 50
        worksheet.column_dimensions['E'].width = 60

    print(f"结果已保存到: {output_path}")


def main():
    print("=" * 60)
    print("商品评论痛点需求分析工具")
    print("=" * 60)

    # 初始化LLM客户端
    print("\n[1/4] 初始化LLM客户端...")
    llm = LLMClient(
        model="qwen-plus",
        temperature=0.3,
    )

    # 连接数据库
    print("[2/4] 连接数据库...")
    try:
        conn = get_db_connection()
        print("数据库连接成功")
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    # 获取随机50个商品
    print("[3/4] 获取商品列表...")
    try:
        product_ids = fetch_random_products(conn, limit=50)
        print(f"获取到 {len(product_ids)} 个商品")
    except Exception as e:
        print(f"获取商品失败: {e}")
        conn.close()
        return

    # 分析每个商品的评论
    print("[4/4] 分析商品评论...")
    results = []

    for i, product_id in enumerate(product_ids, 1):
        print(f"  正在分析商品 {i}/{len(product_ids)}: {product_id}")
        try:
            comments_df = fetch_comments_by_product(conn, product_id)
            if len(comments_df) == 0:
                print(f"    - 无评论，跳过")
                continue

            result = analyze_product_comments(llm, product_id, comments_df)
            results.append(result)
            print(f"    - 完成，共 {result['comment_count']} 条评论")
        except Exception as e:
            print(f"    - 分析失败: {e}")
            results.append({
                'product_id': product_id,
                'comment_count': 0,
                'pain_points': f'处理失败: {str(e)}',
                'user_needs': '',
                'summary': ''
            })

    conn.close()

    # 保存结果
    if results:
        output_path = os.path.join(os.path.dirname(__file__), '评论分析结果.xlsx')
        save_to_excel(results, output_path)
        print(f"\n分析完成！共处理 {len(results)} 个商品")
    else:
        print("\n没有分析结果")


if __name__ == "__main__":
    main()
