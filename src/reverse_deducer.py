"""
Reverse Deduct 模块
- 输入: SQL + 可选数据库 mschema
- 输出: 结构化自然语言 Gold Query 风格
"""

from typing import Optional
import openai
import os
import json

# 使用环境变量配置 OpenAI 客户端
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


reverse_system_prompt = """You are a SQL analysis expert. Given an NLQ, a SQL query, and the database schema, reverse-deduct a concise Gold Query style that semantically matches the SQL. Assess its consistency with the NLQ, briefly explain any differences, and provide actionable SQL fixes if needed."""
reverse_system_prompt_NLQ = """You are a SQL analysis expert. Given an NLQ, a SQL query, and the database schema, reverse-deduct a NLQ that semantically matches the SQL. Assess its consistency with the NLQ, briefly explain any differences, and provide actionable SQL fixes if needed."""

reverse_instruction_prompt = """
**Deduction Rules:**
1. Produce a single-line, natural-language Gold Query that describes the exact semantics of the SQL, and the consistency with the given NLQ with certain reason as feedback, and provide brief actionable SQL fix suggestions if necessary.
2. Follow the Gold Query structure strictly:
   - "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]".
   - For subqueries: Enclose the subquery description in parentheses () to enhance hierarchical structure, e.g., "where the value is in the result of (subquery description)"
   - For join operations: Enclose each pair of join operations in parentheses () to enhance structural clarity, e.g., "from ((table1 join table2 on condition) join table3 on condition)"
   - For conditions: Express them as naturally as possible, not as SQL fragments, e.g, use "where the value is greater than 30" instead of "where value > 30".
3. Be unambiguous: mention table/column names and operations explicitly; keep attribute order consistent with the SELECT list.
4. Use natural wording for conditions (e.g., "greater than 30" instead of "> 30").
5. Output ONLY the Gold Query sentence without explanations.
"""

# 单次调用：同时返回 reverse query 与对齐反馈 JSON
reverse_with_feedback_instruction_prompt = """
**Deduction Rules:**
1. Reverse-deduct a natural language question Gold Query that describes the exact semantics of the SQL, and evaluate whether the reverse-deduced Gold Query is semantically equivalent to the given NLQ.
2. Structured Format: Follow the Gold Query structure:
   - "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]".
3. Unambiguity: mention table/column names and operations explicitly; keep attribute order consistent with the SELECT list.
4. Natural Language: maintain natural English flow while using precise database terminology.

**NLQ:**{question}
**Database Schema:**{db_schema}
**SQL:**{sql}

Return STRICT JSON only, no extra text, with this schema:
```json
{{
  "reverse_query": "one-line natural-language Gold Query",
  "feedback": {{
    "consistency_with_given_question": "true or false",
    "reasoning": "brief reasoning",
    "fixes": ["short, actionable step 1", "step 2", "step 3"]
    }}
}}
```
"""

reverse_with_feedback_instruction_prompt_NLQ = """
**Deduction Rules:**
1. Reverse-deduct a natural language question Gold Query that describes the exact semantics of the SQL, and evaluate whether the reverse-deduced Gold Query is semantically equivalent to the given NLQ.
2. Structured Format: Follow the Gold Query structure:
   - "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]".
3. Unambiguity: mention table/column names and operations explicitly; keep attribute order consistent with the SELECT list.
4. Natural Language: maintain natural English flow while using precise database terminology.

**NLQ:**{question}
**Database Schema:**{db_schema}
**SQL:**{sql}

Return STRICT JSON only, no extra text, with this schema:
```json
{{
  "reverse_query": "one-line natural-language Gold Query",
  "feedback": {{
    "consistency_with_given_question": "true or false",
    "reasoning": "brief reasoning",
    "fixes": ["short, actionable step 1", "step 2", "step 3"]
    }}
}}
```
"""
class ReverseDeducer:
    def __init__(self, model_name: str = "deepseek-v3") -> None:
        self.model_name = model_name

    def deduce_only_reverse_query(self, sql: str, db_schema: Optional[str] = "") -> Optional[str]:
        try:
            prompt = reverse_instruction_prompt.format(db_schema=db_schema, sql=sql)
            system_prompt = reverse_system_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content.replace("\n", " ").strip()
        except Exception as e:
            print(f"Reverse deduction error: {e}")
            return None

    def deduce_query_and_feedback(self, sql: str, question: str, db_schema: Optional[str] = "") -> Optional[dict]:
        try:
            prompt = reverse_with_feedback_instruction_prompt_NLQ.format(
                db_schema=db_schema,
                sql=sql,
                question=question,
            )
            system_prompt = reverse_system_prompt_NLQ
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            text = resp.choices[0].message.content.replace("\n", " ").strip()
            if text.startswith("```"):
                start = text.find("{")
                end = text.rfind("}")
                text = text[start:end+1]
            return json.loads(text)
        except Exception as e:
            print(f"Reverse deduction (with feedback) error: {e}")
            return None


def create_reverse_deducer(model_name: str = "deepseek-v3") -> ReverseDeducer:
    return ReverseDeducer(model_name)


