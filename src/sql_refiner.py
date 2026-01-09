"""
SQL Refiner 模块
- 基于 general_execution_feedback 和 semantic_consistency_feedback 对生成的SQL进行优化
- 与 query_refiner 分离，专门处理SQL层面的优化
- 包含字符串匹配后处理功能，用于修正大小写不匹配问题
"""

from typing import Dict, List, Optional, Union
import json
import openai
import os
import re
import sqlite3

# OpenAI客户端配置
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# System Prompt：角色与目标
sql_refine_system_prompt = """You are a SQL Refiner. Your job is to refine and optimize SQL queries based on execution feedback and semantic consistency feedback."""

# Instruction Prompt：输入与输出要求（包含semantic feedback）
sql_refine_instruction_prompt_with_semantic = """
**Original Question: {original_question}
- Current SQL Query: {current_sql}
- Database Schema: {db_schema}
- Semantic Consistency Feedback (JSON): {semantic_feedback}
- Execution Feedback (JSON): {execution_feedback}

**Refine Goals:**
1. Fix all issues mentioned in the feedback while preserving the original intent.
2. Optimize SQL correctness based on execution results.
3. Ensure the refined SQL is syntactically correct and executable.
4. Maintain semantic equivalence with the original question.

Output:
- Return ONLY the refined SQL with no explanations.
- The output SQL should be a clean, executable SQL statement and protected by ```sql and ```.
"""

# Instruction Prompt：输入与输出要求（仅包含execution feedback）
sql_refine_instruction_prompt_execution_only = """
**Original Question: {original_question}
- Current SQL Query: {current_sql}
- Database Schema: {db_schema}
- Execution Feedback (JSON): {execution_feedback}

**Refine Goals:**
1. Fix all issues mentioned in the execution feedback while preserving the original intent.
2. Optimize SQL correctness based on execution results.
3. Ensure the refined SQL is syntactically correct and executable.
4. Maintain semantic equivalence with the original question.

Output:
- Return ONLY the refined SQL with no explanations.
- The output SQL should be a clean, executable SQL statement and protected by ```sql and ```.
"""


class SQLRefiner:
    """SQL Refiner 实现类"""

    def __init__(self, model_name: str = "deepseek-v3") -> None:
        self.model_name = model_name

    def refine(self,
               original_question: str,
               current_sql: str,
               db_schema: str,
               execution_feedback: Optional[Union[Dict, str]],
               semantic_feedback: Optional[Dict] = None
               ) -> Optional[str]:
        """基于反馈进行SQL优化，返回优化后的SQL"""
        try:
            # 准备反馈数据
            if not execution_feedback:
                execution_feedback_json = "{}"
            elif isinstance(execution_feedback, str):
                execution_feedback_json = json.dumps({"error_message": execution_feedback}, ensure_ascii=False, separators=(",", ":"))
            else:
                execution_feedback_json = json.dumps(execution_feedback, ensure_ascii=False, separators=(",", ":"))
            
            # 根据是否有semantic_feedback选择不同的prompt
            if semantic_feedback:
                semantic_feedback_json = json.dumps(semantic_feedback, ensure_ascii=False, separators=(",", ":"))
                prompt = sql_refine_instruction_prompt_with_semantic.format(
                    original_question=original_question,
                    current_sql=current_sql,
                    db_schema=db_schema,
                    execution_feedback=execution_feedback_json,
                    semantic_feedback=semantic_feedback_json
                )
            else:
                prompt = sql_refine_instruction_prompt_execution_only.format(
                    original_question=original_question,
                    current_sql=current_sql,
                    db_schema=db_schema,
                    execution_feedback=execution_feedback_json
                )
            
            messages = [
                {"role": "system", "content": sql_refine_system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            refined_sql = response.choices[0].message.content.replace("\n", " ").strip()
            
            # 清理输出，移除可能的markdown格式
            refined_sql = refined_sql.replace("```sql", "").replace("```", "").strip()
            
            return refined_sql
            
        except Exception as e:
            print(f"SQL refine error: {e}")
            return None

    def _print_event(self, event_type: str, payload: Dict) -> None:
        """以结构化JSON打印事件日志。"""
        try:
            print(json.dumps({"event": event_type, **payload}, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            # 兜底：即便JSON失败也不影响主流程
            pass

    def refine_string_matches_only(self,
                                  current_sql: str,
                                  db_path: str,
                                  original_question: str
                                  ) -> tuple[Optional[str], bool]:
        """仅进行字符串匹配后处理，不进行其他refine"""
        # 检查SQL是否有效
        if not is_valid_sql(current_sql, db_path):
            print("SQL无效，跳过字符串匹配后处理")
            return current_sql, False
        
        # 执行字符串匹配后处理
        refined_sql, was_refined = post_process_string_matches(current_sql, db_path, original_question)
        
        if was_refined:
            print("应用了字符串匹配后处理")
        
        return refined_sql, was_refined

    def refine_with_string_postprocess(self,
                                     original_question: str,
                                     current_sql: str,
                                     db_schema: str,
                                     db_path: str,
                                     execution_feedback: Optional[Union[Dict, str]],
                                     semantic_feedback: Optional[Dict] = None
                                     ) -> Optional[str]:
        """先进行字符串匹配后处理，再进行正常的反馈refine"""
        # 首先进行字符串匹配后处理
        string_processed_sql, was_string_refined = self.refine_string_matches_only(
            current_sql=current_sql,
            db_path=db_path,
            original_question=original_question
        )
        
        # 然后进行正常的反馈refine
        final_sql = self.refine(
            original_question=original_question,
            current_sql=string_processed_sql,
            db_schema=db_schema,
            execution_feedback=execution_feedback,
            semantic_feedback=semantic_feedback
        )
        
        return final_sql


def extract_string_matches_from_sql(sql):
    """从SQL中提取字符串匹配条件"""
    string_matches = []
    
    # 匹配所有 = 'string' 格式的字符串，包括WHERE、AND、OR、JOIN等条件
    pattern = r"=\s*'([^']+)'"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    for match in matches:
        string_matches.append(match)
    
    # 匹配 IN ('string1', 'string2') 格式的字符串
    pattern = r"IN\s*\(\s*'([^']+)'"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    for match in matches:
        string_matches.append(match)
    
    # 匹配 LIKE '%string%' 格式的字符串
    pattern = r"LIKE\s+'([^']+)'"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    for match in matches:
        string_matches.append(match)
    
    return string_matches

def check_string_in_database(db_path, sql, string_value):
    """检查数据库中的实际字符串值"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有表的列信息
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            # 获取表的所有列
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for column in columns:
                column_name = column[1]
                # 检查该列中是否有与查询字符串相似的值（忽略大小写）
                cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE LOWER({column_name}) = LOWER(?)", (string_value,))
                results = cursor.fetchall()
                
                if results:
                    # 找到匹配项，返回实际的字符串值
                    actual_values = [row[0] for row in results]
                    conn.close()
                    return actual_values
        
        conn.close()
        return None
    except Exception as e:
        print(f"检查数据库时出错: {e}")
        return None

def refine_sql_with_correct_strings(sql, string_corrections):
    """使用正确的字符串值重新生成SQL"""
    refined_sql = sql
    for original, correct in string_corrections.items():
        # 替换SQL中的字符串值
        refined_sql = re.sub(f"'{original}'", f"'{correct}'", refined_sql, flags=re.IGNORECASE)
    return refined_sql

def post_process_string_matches(sql, db_path, question):
    """后处理字符串匹配，检查并修正大小写不匹配"""
    string_matches = extract_string_matches_from_sql(sql)
    
    if not string_matches:
        return sql, False
    
    string_corrections = {}
    needs_refinement = False
    
    for string_value in string_matches:
        # 检查数据库中的实际值
        actual_values = check_string_in_database(db_path, sql, string_value)
        
        if actual_values:
            # 找到实际值，检查是否需要修正
            for actual_value in actual_values:
                if actual_value != string_value:
                    string_corrections[string_value] = actual_value
                    needs_refinement = True
                    print(f"发现字符串不匹配: '{string_value}' -> '{actual_value}'")
                    break
    
    if needs_refinement:
        # 使用正确的字符串值重新生成SQL
        refined_sql = refine_sql_with_correct_strings(sql, string_corrections)
        print(f"原始SQL: {sql}")
        print(f"修正后SQL: {refined_sql}")
        return refined_sql, True
    
    return sql, False

def is_valid_sql(sql, db_path):
    """检查SQL是否有效（语法正确且可执行）"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 尝试执行SQL，如果成功则说明SQL有效
        cursor.execute(sql)
        conn.close()
        return True
    except Exception as e:
        print(f"SQL验证失败: {e}")
        return False

def create_sql_refiner(model_name: str = "deepseek-v3") -> SQLRefiner:
    """创建SQL优化器实例"""
    return SQLRefiner(model_name)
