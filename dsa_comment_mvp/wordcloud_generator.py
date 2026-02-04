"""
评论词云生成工具
生成心形词云图（只保留名词和形容词）
"""
import pyodbc
import jieba
import jieba.posseg as pseg  # 词性标注
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
from collections import Counter
import numpy as np
from PIL import Image

# 数据库配置
SQL_SERVER_CONFIG = {
    'SERVER': '121.46.231.162,51433',
    'DATABASE': 'Compass_BP',
    'UID': 'compassBase',
    'PWD': 'hSG55#@$8D2AHJF;',
    'USE_WINDOWS_AUTH': False,
}

# 停用词（可根据需要扩展）
STOP_WORDS = {
    '的', '了', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看',
    '好', '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么',
    '怎么', '如果', '因为', '所以', '但是', '而且', '或者', '虽然', '然后',
    '可以', '可能', '应该', '已经', '还是', '还有', '比较', '非常', '特别',
    '真的', '确实', '觉得', '感觉', '东西', '东西', '产品', '商品', '购买',
    '买', '收到', '发货', '快递', '物流', '包装', '下单', '订单', '客服',
    '店家', '卖家', '一直', '之前', '以后', '现在', '时候', '使用', '用',
}


def get_db_connection():
    """创建数据库连接"""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER_CONFIG['SERVER']};"
        f"DATABASE={SQL_SERVER_CONFIG['DATABASE']};"
        f"UID={SQL_SERVER_CONFIG['UID']};"
        f"PWD={SQL_SERVER_CONFIG['PWD']};"
    )
    return pyodbc.connect(conn_str)


def fetch_comments(conn, limit: int = 1000) -> list:
    """从数据库随机获取评论"""
    query = f"""
        SELECT TOP {limit} content
        FROM dsa_jd_comments_detail
        WHERE content IS NOT NULL AND LEN(content) > 5
        ORDER BY NEWID()
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall() if row[0]]


def segment_text(texts: list) -> str:
    """对评论进行分词，只保留名词和形容词，去重"""
    # 名词词性标签：n 名词, nr 人名, ns 地名, nt 机构名, nz 其他专名
    # 形容词词性标签：a 形容词, ad 副形词, an 名形词
    noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'ng'}  # 名词类
    adj_tags = {'a', 'ad', 'an', 'ag'}  # 形容词类

    allowed_tags = noun_tags | adj_tags

    word_set = set()  # 使用set去重

    for text in texts:
        words = pseg.lcut(text)  # 带词性标注的分词
        for word, flag in words:
            # 过滤条件：词性在允许范围内，长度>1，不在停用词中，不是纯数字
            if (flag in allowed_tags and
                len(word) > 1 and
                word not in STOP_WORDS and
                not word.isdigit()):
                word_set.add(word)

    return ' '.join(word_set)


def create_heart_mask(width=800, height=800):
    """生成心形mask"""
    x = np.linspace(-1.5, 1.5, width)
    y = np.linspace(-1.5, 1.5, height)
    X, Y = np.meshgrid(x, y)

    # 心形方程: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0
    # 翻转Y轴使心形正向
    Y = -Y
    heart = (X**2 + Y**2 - 1)**3 - X**2 * Y**3

    # 创建mask：心形内部为白色(255)，外部为黑色(0)
    # WordCloud要求：白色区域不绘制，非白色区域绘制
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[heart <= 0] = 255  # 心形内部

    # WordCloud需要的是：白色(255)为不绘制区域
    # 所以需要反转：心形内部为非白色，外部为白色
    mask_inverted = 255 - mask

    # 转换为RGB格式
    mask_rgb = np.stack([mask_inverted]*3, axis=-1)

    return mask_rgb


def generate_wordcloud(text: str, output_path: str, colormap: str = 'Reds'):
    """生成心形词云图"""
    # 尝试使用系统中文字体
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
        'C:/Windows/Fonts/simhei.ttf',    # 黑体
        'C:/Windows/Fonts/simsun.ttc',    # 宋体
    ]

    font_path = None
    for fp in font_paths:
        if os.path.exists(fp):
            font_path = fp
            break

    if not font_path:
        print("警告: 未找到中文字体，词云可能显示乱码")

    # 生成心形mask
    heart_mask = create_heart_mask(800, 800)

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=800,
        background_color='white',
        max_words=200,
        max_font_size=150,
        random_state=42,
        colormap=colormap,
        mask=heart_mask,
        contour_width=2,
        contour_color='red',
    )

    wc.generate(text)

    # 保存图片
    wc.to_file(output_path)
    print(f"词云图已保存: {output_path}")

    # 显示图片
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('商品评论词云（名词+形容词）', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_display.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    return wc


def main():
    print("=" * 60)
    print("评论词云生成工具 - 心形词云（名词+形容词）")
    print("=" * 60)

    # 连接数据库
    print("\n[1/4] 连接数据库...")
    try:
        conn = get_db_connection()
        print("数据库连接成功")
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    # 获取评论
    print("[2/4] 获取评论数据...")
    try:
        comments = fetch_comments(conn, limit=1000)
        print(f"获取到 {len(comments)} 条评论")
    except Exception as e:
        print(f"获取评论失败: {e}")
        conn.close()
        return

    conn.close()

    # 分词（只保留名词和形容词，去重）
    print("[3/4] 分词处理（提取名词+形容词，去重）...")
    text = segment_text(comments)
    word_count = len(text.split())
    print(f"分词完成，共 {word_count} 个不重复词汇")

    # 生成心形词云
    print("[4/4] 生成心形词云...")
    output_path = os.path.join(os.path.dirname(__file__), '评论词云_心形.png')
    generate_wordcloud(text, output_path)

    print("\n完成！")


if __name__ == "__main__":
    main()
