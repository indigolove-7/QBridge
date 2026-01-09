"""
问题重写评估模块
包含多级评估和rubber duck检查功能
"""

import openai
import os
import json
from typing import Dict, List, Tuple, Optional

# OpenAI客户端配置
# 使用环境变量配置 OpenAI 客户端，避免硬编码
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 评估提示词 - 优化版本，减少token使用
review_system_prompt = """You are a professional question rewriting quality reviewing specialist. Evaluate a single rewritten question across following dimensions using rubber duck checking.

Dimensions:
1. STRUCTURED FORMAT - Does it follow the structured format for Gold Query?
   - Base pattern: "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]"
2. ORDERING CORRECTNESS - Does the rewritten question specify the correct ordering (e.g., asc/desc) that matches the original question's intent, especially for superlative queries (e.g., "fewest", "most", "highest", "lowest")? Check for misinterpretation of order direction.
3. GROUP BY USAGE - Only use GROUP BY when it is truly necessary; avoid excessive grouping.
4. SET OPERATION LOGIC - Check if the question contains keywords such as "either...or" (should map to UNION), "both...and" (should map to INTERSECT), and "except"/"not in" (should map to EXCEPT or NOT IN depending on context). Ensure the rewritten question correctly reflects the intended set operation or aggregation logic, and matches the original meaning.
5. JOIN CONDITIONS - By default, do not specify explicit join-on columns (including foreign keys); only specify join-on columns when omitting them would cause ambiguity or confusion. Only use join instead of inner join or left join.
6. UNAMBIGUITY - Are table names, column names, and functions clearly specified?
7. DB UTILIZATION - Does the rewritten question utilize the schema effectively but not excessively? Also, pay attention to case sensitivity issues between the schema and the original question.
8. NATURAL LANGUAGE - Does the rewritten question maintain natural English flow while using precise database terminology?

Use rubber duck checking: briefly explain your reasoning per dimension, then provide brief fix suggestions if needed."""

review_instruction_prompt = """
**Schema:** {db_schema}
**Original Question:** {original_question}
**Rewritten Question:** {rewritten_question}

Output JSON format only:
```json
{{
  "rubber_duck_feedback": {{
    "structured_format": "...",
    "ordering_correctness": "...",
    "group_by_usage": "...",
    "set_operation_logic": "...",
    "join_conditions": "...",
    "unambiguity": "...",
    "db_utilization": "...",
    "natural_language": "..."
  }},
  "fixes": ["short, actionable step 1", "step 2", "step 3"]
}}
```
"""

batch_review_instruction_prompt = """
**Original Question:** {original_question}
**Schema:** {db_schema}
**Rewritten Questions:**{rewritten_questions}

Return STRICT JSON only, no extra text, with this schema:
```json
{{
  "rubber_duck_feedback": {{
    "structured_format": "brief reasoning ...",
    "ordering_correctness": "brief reasoning ...",
    "group_by_usage": "brief reasoning ...",
    "set_operation_logic": "brief reasoning ...",
    "join_conditions": "brief reasoning ...",
    "unambiguity": "brief reasoning ...",
    "db_utilization": "brief reasoning ...",
    "natural_language": "brief reasoning ..."
  }},
  "fixes": ["short, actionable step 1", "step 2", "step 3"]
}}
```
"""


class RewriteReviewer:
    """问题重写评估器类"""
    
    def __init__(self, model_name="deepseek-v3"):
        """初始化评估器"""
        self.model_name = model_name
    
    def _call_api(self, messages, temperature=0.0):
        """调用OpenAI API"""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.replace("\n", " ")
        except Exception as e:
            print(f"API调用错误: {e}")
            return None
    
    def _parse_json_response(self, response_text):
        """解析JSON响应"""
        try:
            # 尝试提取JSON部分
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            return json.loads(json_text)
        except Exception as e:
            print(f"JSON解析错误: {e}")
            return None
    
    def evaluate_single(self, original_question: str, rewritten_question: str, db_schema: str) -> Optional[Dict]:
        """评估单个重写问题"""
        prompt = review_instruction_prompt.format(
            original_question=original_question,
            rewritten_question=rewritten_question,
            db_schema=db_schema
        )

        messages = [
            {"role": "system", "content": review_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages)
        if not response:
            return None
        
        result = self._parse_json_response(response)
        
        return result
    
    def evaluate_batch(self, original_question: str, rewritten_questions: List[str], db_schema: str) -> Optional[Dict]:
        """批量评估多个重写问题"""
        questions_text = ""
        for i, question in enumerate(rewritten_questions, 1):
            questions_text += f"**Question {i}:** {question}\n\n"
        
        prompt = batch_review_instruction_prompt.format(
            original_question=original_question,
            db_schema=db_schema,
            rewritten_questions=questions_text
        )
        
        messages = [
            {"role": "system", "content": review_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages)
        if not response:
            return None
        
        result = self._parse_json_response(response)
        
        return result
    
    def get_quality_summary(self, evaluation_result: Dict) -> str:
        """生成反馈摘要（不包含打分，仅提炼rubber duck反馈与修复建议）"""
        if not evaluation_result:
            return "反馈生成失败"

        rd = evaluation_result.get("rubber_duck_feedback", {})
        fixes = evaluation_result.get("fixes", [])

        lines = []
        if rd:
            lines.append("Rubber duck 反馈：")
            if rd.get("structured_format"):
                lines.append(f"- 结构化格式: {rd['structured_format']}")
            if rd.get("unambiguity"):
                lines.append(f"- 明确性: {rd['unambiguity']}")
            if rd.get("join_conditions"):
                lines.append(f"- JOIN 条件: {rd['join_conditions']}")
            if rd.get("ordering_correctness"):
                lines.append(f"- 排序正确性: {rd['ordering_correctness']}")
            if rd.get("group_by_usage"):
                lines.append(f"- GROUP BY 使用: {rd['group_by_usage']}")
            if rd.get("set_operation_logic"):
                lines.append(f"- 关键词与集合操作逻辑: {rd['set_operation_logic']}")
            if rd.get("db_utilization"):
                lines.append(f"- 数据库利用率: {rd['db_utilization']}")
            if rd.get("natural_language"):
                lines.append(f"- 自然语言: {rd['natural_language']}")
        if fixes:
            lines.append("\n建议的修复步骤：")
            for i, s in enumerate(fixes, 1):
                lines.append(f"{i}. {s}")

        return "\n".join(lines)


def create_reviewer(model_name="deepseek-v3"):
    """创建评估器实例的工厂函数"""
    return RewriteReviewer(model_name)


# 便捷函数
def quick_evaluate(original_question: str, rewritten_question: str, db_schema: str, model_name="deepseek-v3") -> Optional[Dict]:
    """快速评估单个重写问题"""
    reviewer = create_reviewer(model_name)
    return reviewer.evaluate_single(original_question, rewritten_question, db_schema)