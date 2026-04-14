"""
聊天数据处理器 - 从聊天记录中提取问答对
"""
import json
import re
from typing import List, Dict, Optional


def _safe_print(message: str) -> None:
    """在不同终端编码下尽量安全地输出日志。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("gbk", errors="replace").decode("gbk"))


class ChatDataProcessor:
    """聊天数据清洗与问答对提取"""

    # 问答对分类关键词映射
    CATEGORY_KEYWORDS = {
        "价格": ["价格", "多少钱", "优惠", "折扣", "券", "满减", "便宜", "贵", "划算", "活动价"],
        "物流": ["发货", "快递", "运费", "包邮", "到货", "几天", "物流", "配送", "寄"],
        "售后": ["退", "换", "质量", "售后", "维修", "保修", "赔偿", "运费险", "投诉"],
        "产品": ["材质", "面料", "尺码", "大小", "颜色", "款式", "规格", "参数", "成分"],
        "库存": ["有货", "库存", "预定", "预售", "补货", "现货", "缺货", "到货"],
    }

    # 需要清洗的文本模式
    CLEAN_PATTERNS = [
        (r"[^\u4e00-\u9fa5a-zA-Z0-9%，。！？、~～·\-\+\d元件个件套包箱]", ""),  # 移除特殊字符
        (r"\s+", " "),  # 多余空白
        (r"~+", "~"),  # 多余波浪号
        (r"[。！]{2,}", "。"),  # 多余标点
    ]

    def __init__(self):
        """初始化数据处理器"""
        self._qa_pairs: List[Dict] = []

    def clean_text(self, text: str) -> str:
        """
        清洗单条文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        text = text.strip()
        for pattern, replacement in self.CLEAN_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text.strip()

    def classify_question(self, question: str) -> str:
        """
        对问题进行分类

        Args:
            question: 客户问题文本

        Returns:
            分类标签
        """
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question:
                    return category
        return "通用"

    def extract_qa_pairs(self, chat_records: List[Dict]) -> List[Dict]:
        """
        从聊天记录中提取问答对

        Args:
            chat_records: 聊天记录列表，每条记录包含 role 和 content 字段

        Returns:
            问答对列表，每条包含 question, answer, category 字段
        """
        qa_pairs: List[Dict] = []
        i = 0

        def role_of(msg: Dict) -> str:
            return (msg.get("role") or "").strip().lower()

        while i < len(chat_records):
            current = chat_records[i]
            if role_of(current) != "customer":
                i += 1
                continue

            question = self.clean_text(current.get("content", ""))
            if not question or len(question) < 2:
                i += 1
                continue

            # 从当前 customer 往后找第一条 merchant 回复；允许中间夹杂其它角色/空内容
            j = i + 1
            answer_parts: List[str] = []
            found_merchant = False

            while j < len(chat_records):
                msg = chat_records[j]
                r = role_of(msg)
                content = self.clean_text(msg.get("content", ""))

                # 下一条客户出现：结束本轮查找（不跨客户配对）
                if r == "customer":
                    break

                if r == "merchant" and content:
                    found_merchant = True
                    answer_parts.append(content)
                else:
                    # 其它角色/系统提示/空行：忽略
                    pass

                # 一旦找到商家回复后，继续合并后续连续的商家回复；
                # 如果后面再出现客户，就会在上面 break
                j += 1

            if found_merchant:
                answer = self.clean_text(" ".join(answer_parts))
                if answer and len(answer) >= 4:
                    category = self.classify_question(question)
                    qa_pairs.append(
                        {
                            "question": question,
                            "answer": answer,
                            "category": category,
                        }
                    )

            # 指针移动到本轮结束位置：如果遇到下一条 customer，就从那里继续
            i = j

        self._qa_pairs = qa_pairs
        return qa_pairs

    def save_to_json(self, qa_pairs: List[Dict], filepath: str) -> None:
        """
        将问答对保存为JSON文件

        Args:
            qa_pairs: 问答对列表
            filepath: 保存路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        _safe_print(f"  [OK] 问答对已保存至 {filepath}")

    def load_from_json(self, filepath: str) -> List[Dict]:
        """
        从JSON文件加载问答对

        Args:
            filepath: JSON文件路径

        Returns:
            问答对列表
        """
        with open(filepath, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        self._qa_pairs = qa_pairs
        return qa_pairs

    def get_statistics(self, qa_pairs: Optional[List[Dict]] = None) -> Dict:
        """
        获取问答对统计信息

        Args:
            qa_pairs: 问答对列表（默认使用已提取的）

        Returns:
            统计信息字典
        """
        data = qa_pairs or self._qa_pairs
        if not data:
            return {"total": 0, "categories": {}}

        category_count = {}
        for qa in data:
            cat = qa.get("category", "未知")
            category_count[cat] = category_count.get(cat, 0) + 1

        return {
            "total": len(data),
            "categories": category_count,
        }
