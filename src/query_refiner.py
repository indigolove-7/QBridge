"""
Query Refiner 模块
- 接收原问题、改写问题、schema 以及 reviewer 的反馈（rubber_duck_feedback + fix_suggestions）
- 产出最终的 Gold Query，用于下游 SQL 生成
"""

from typing import Dict, List, Optional
import json
import openai
import os

# OpenAI客户端配置（与其他模块一致）
# 使用环境变量配置 OpenAI 客户端，避免硬编码
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# System Prompt：角色与目标
refine_system_prompt = """You are a professional Query Refining Specialist. I will provide an original question, a rewritten question, corresponding database schema, and a reviewer feedback. Your task is to refine the rewritten question guided by the reviewer feedback."""

# Instruction Prompt：输入与输出要求
refine_instruction_prompt = """
**Refine Goals:**
1. FEEDBACK GUIDANCE: Fix all issues mentioned in the reviewer feedback while preserving semantic equivalence with the original question.
2. UNAMBIGUOUS: Explicitly specify aggregations (count, sum, avg, max, min) and comparisons, and distinguish between column names and aggregation functions
3. STRUCTURED FORMAT: Still follow this exact Gold Query structure as the rewritten question:
   - Base pattern: "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]"
4. NATURAL LANGUAGE: Maintain natural English flow while using precise database terminology, not as SQL fragments.
5. OUTPUT: Return only the refined rewritten question with no explanations, if there is no issue, return the rewritten question directly.

**Schema:**
{db_schema}
**Original Question:**
{original_question}
**Rewritten Question:**
{rewritten_question}

**Reviewer Feedback (JSON):**
{reviewer_feedback}

Return only the refined rewritten question with no explanations, if there is no issue, return the rewritten question directly.
"""


class QueryRefiner:
    """Query Refiner 实现类"""

    def __init__(self, model_name: str = "deepseek-v3") -> None:
        self.model_name = model_name

    def refine(self,
               original_question: str,
               rewritten_question: str,
               db_schema: str,
               reviewer_feedback: Dict,
               execution_feedback: Optional[Dict] = None) -> Optional[str]:
        """基于反馈进行refine，返回最终Gold Query"""
        try:
            feedback_json = json.dumps(reviewer_feedback, ensure_ascii=False, separators=(",", ":"))
            
            # 构建包含执行反馈的提示
            execution_feedback_text = ""
            if execution_feedback:
                execution_feedback_json = json.dumps(execution_feedback, ensure_ascii=False, separators=(",", ":"))
                execution_feedback_text = f"\n\n- Execution Feedback (JSON):\n{execution_feedback_json}"
            
            prompt = refine_instruction_prompt.format(
                original_question=original_question,
                db_schema=db_schema,
                rewritten_question=rewritten_question,
                reviewer_feedback=feedback_json,
            ) + execution_feedback_text
            messages = [
                {"role": "system", "content": refine_system_prompt},
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Refine error: {e}")
            return None


def create_query_refiner(model_name: str = "deepseek-v3") -> QueryRefiner:
    return QueryRefiner(model_name)


# 简单便捷函数
def quick_refine(original_question: str,
                 rewritten_question: str,
                 db_schema: str,
                 reviewer_feedback: Dict,
                 model_name: str = "deepseek-v3",
                 execution_feedback: Optional[Dict] = None) -> Optional[str]:
    refiner = create_query_refiner(model_name)
    return refiner.refine(original_question, rewritten_question, db_schema, reviewer_feedback, execution_feedback)
