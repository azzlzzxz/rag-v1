"""
意图识别模块 —— 此处我们基于规则 进行简单实现
其次复杂场景下，可以使用通用LLM进行意图识别或者使用微调的模型进行
"""

class IntentRecognizer:
    """意图识别器"""
    def __init__(self):
        self.intents_rules = {
            "知识查询": ["什么是", "介绍", "说明", "定义", "含义", "解释", "描述"],
            "操作指导": ["如何", "怎样", "怎么", "步骤", "方法", "流程", "操作", "指导"],
            "政策咨询": ["政策", "规定", "制度", "办法", "条例", "通知", "文件"],
            "对比分析": ["区别", "不同", "比较", "对比", "优缺点", "哪个好"],
            "数据统计": ["多少", "数量", "比例", "占比", "统计", "数据"],
            "故障排查": ["错误", "失败", "问题", "异常", "报错", "解决", "修复"],
        }
    
    def recognize(self, query: str) -> str:
        """
        识别用户意图 (基于规则)
        
        Args:
            query: 用户输入的查询语句
            
        Returns:
            str: 识别到的意图类型
        """
        if not query:
            return "通用问答"
        
        for intent, keywords in self.intents_rules.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return "通用问答"

    def get_all_intents(self) -> list:
        """获取所有意图类型"""
        return list(self.intents_rules.keys()) + ["通用问答"]
