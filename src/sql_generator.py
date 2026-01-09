"""
SQL Generator 模块
- 基于问题、schema和提示生成SQL查询
- 封装SQL生成逻辑，提供统一的接口
"""

import openai
import os
import re
from typing import List, Dict, Optional
from sql_post_process import fix_select_column

# OpenAI客户端配置
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# SQL生成提示模板
sql_gen_chat_prompt = [
    {
        "role": "system",
        "content": "You are now an excellent SQL writer, first I'll give you some tips and examples, and I need you to remember the tips, and do not make same mistakes."
    },
    {
        "role": "user",
        "content": """Tips 1: 
Question: Which A has most number of B?
Gold SQL: select A from B group by A order by count ( * ) desc limit 1;
Notice that the Gold SQL doesn't select COUNT(*) because the question only wants to know the A and the number should be only used in ORDER BY clause, there are many questions asks in this way, and I need you to remember this in the the following questions."""
    },
    {
        "role": "assistant",
        "content": "Thank you for the tip! I'll keep in mind that when the question only asks for a certain field, I should not include the COUNT(*) in the SELECT statement, but instead use it in the ORDER BY clause to sort the results based on the count of that field."
    },
    {
        "role": "user",
        "content": """Tips 2: 
Don't use "IN", "OR", "LEFT JOIN" as it might cause extra results, use "INTERSECT" or "EXCEPT" instead, and remember to use "DISTINCT" or "LIMIT" when necessary.
For example, 
Question: Who are the A who have been nominated for both B award and C award?
Gold SQL should be: select A from X where award = 'B' intersect select A from X where award = 'C';"""
    },
    {
        "role": "assistant",
        "content": "Thank you for the tip! I'll remember to use \"INTERSECT\" or \"EXCEPT\" instead of \"IN\", \"OR\", or \"LEFT JOIN\" when I want to find records that match or don't match across two tables. Additionally, I'll make sure to use \"DISTINCT\" or \"LIMIT\" when necessary to avoid repetitive results or limit the number of results returned."
    }
]

# 额外的提示
other2tips = [
    {
        "role": "user",
        "content": """Tips 3: 
When generating SQL, the order of fields in the SELECT clause should be consistent with the order of attributes in the question.
For example,
Question: What are the A and B of all C?
Gold SQL should select A first and then B, the generated SQL should be: SELECT A, B FROM C;"""
    },
    {
        "role": "assistant",
        "content": "Thank you for the tip! I'll remember to select the fields in the same order as the attributes in the question to maintain the consistency."
    },
    {
        "role": "user",
        "content": """Tips 4: 
When SQL involves multiple tables, make sure to use standardized table aliases (like T1, T2, T3) when writing multi-table JOIN queries.
For example,
Question: What are the names and ids of all countries with at least one car maker?
Gold SQL should be: SELECT T1.CountryName, T1.CountryId FROM COUNTRIES AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId = T2.Country GROUP BY T1.CountryId HAVING count(*) >= 1;"""
    },
    {
        "role": "assistant",
        "content": "Thank you for the tip! I'll remember to use standardized table aliases (like T1, T2, T3) when writing multi-table JOIN queries instead of using table names directly. This will improve SQL readability and standardization."
    }
]


class SQLGenerator:
    """SQL生成器实现类"""
    
    def __init__(self, model_name: str = "deepseek-v3"):
        self.model_name = model_name
    
    def generate_sql(self, 
                    input_sequence: str, 
                    max_retries: int = 5,
                    use_additional_tips: bool = False,
                    temperature: float = 1.0) -> Optional[str]:
        """
        生成SQL查询
        
        Args:
            input_sequence: 输入序列（包含问题、schema等）
            max_retries: 最大重试次数
            use_additional_tips: 是否使用额外提示
            temperature: 生成温度（默认1.0，用于单条生成）
            
        Returns:
            生成的SQL字符串，失败时返回None
        """
        processed_sql = None
        for attempt in range(max_retries):
            try:
                # 构建消息
                messages = self._build_messages(input_sequence, use_additional_tips)
                
                # 调用LLM生成SQL
                sql = self._call_llm(messages)
                if sql:
                    # 后处理SQL
                    processed_sql = self._post_process_sql(sql)

                    return processed_sql
                    
            except Exception as e:
                print(f"SQL generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
        return processed_sql
    
    def generate_multiple_sqls(self, 
                              input_sequence: str, 
                              n: int = 1,
                              use_additional_tips: bool = True,
                              temperature: float = 1.0) -> List[str]:
        """
        生成多个SQL查询（用于self-consistency）
        
        Args:
            input_sequence: 输入序列
            n: 生成数量
            use_additional_tips: 是否使用额外提示
            
        Returns:
            SQL字符串列表
        """
        try:
            messages = self._build_messages(input_sequence, use_additional_tips)
            
            if n == 1:
                # 单次生成
                sql = self._call_llm(messages)
                return [sql] if sql else []
            else:
                # 多次生成
                sqls = []
                for _ in range(n):
                    sql = self._call_llm(messages)
                    if sql:
                        processed_sql = self._post_process_sql(sql)
                        sqls.append(processed_sql)
                return sqls
                
        except Exception as e:
            print(f"Multiple SQL generation failed: {e}")
            return []
    
    def _build_messages(self, input_sequence: str, use_additional_tips: bool) -> List[Dict]:
        """构建LLM消息"""
        messages = sql_gen_chat_prompt.copy()
        
        if use_additional_tips:
            messages.extend(other2tips)
        
        messages.append({"role": "user", "content": input_sequence})
        return messages
    
    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        """调用LLM生成SQL，使用默认温度"""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content.replace("\n", " ").strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    def _post_process_sql(self, sql: str) -> str:
        """后处理SQL"""
        # 移除markdown格式 - 更全面的处理
        # 处理各种可能的markdown格式
        sql = sql.replace("```sql\n", "").replace("```sql ", "").replace("```\n", "").replace("```", "")
        
        # 清理SQL字符串，移除多余的空格和换行
        sql = sql.strip()
        
        # 修复各种SELECT相关的问题
        # 1. 修复重复的SELECT
        sql = sql.replace("SELECT SELECT", "SELECT")
        sql = sql.replace("select select", "SELECT")
        sql = sql.replace("select SELECT", "SELECT")
        sql = sql.replace("SELECT select", "SELECT")
        
        # 修复其他可能的SELECT后跟关键字的情况
        sql = sql.replace("SELECT SQL", "SELECT")
        sql = sql.replace("select SQL", "SELECT")
        
        # 确保以SELECT开头（在修复重复SELECT之后）
        if not sql.upper().startswith('SELECT') and not sql.upper().startswith('select'):
            sql = 'SELECT ' + sql
        
        # 修复比较操作符
        sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        
        # 尝试修复SELECT列
        try:
            sql = fix_select_column(sql)
        except Exception as e:
            print(f"fix_select_column error: {e}")
        
        return sql
    
    def generate_with_validation(self, 
                                input_sequence: str, 
                                db_path: str,
                                max_retries: int = 5) -> Optional[str]:
        """
        生成SQL并进行验证
        
        Args:
            input_sequence: 输入序列
            db_path: 数据库路径
            max_retries: 最大重试次数
            
        Returns:
            验证通过的SQL，失败时返回None
        """
        def is_valid(sql, db_path):
            """验证SQL是否有效"""
            import sqlite3
            import re
            
            def replace_cur_year(query: str) -> str:
                return re.sub(
                    "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
                )
            
            def get_cursor_from_path(sqlite_path: str):
                try:
                    if not os.path.exists(sqlite_path):
                        print("Openning a new connection %s" % sqlite_path)
                    connection = sqlite3.connect(sqlite_path)
                except Exception as e:
                    print(sqlite_path)
                    raise e
                connection.text_factory = lambda b: b.decode(errors="ignore")
                cursor = connection.cursor()
                return cursor
            
            def exec_on_db_(sqlite_path: str, query: str):
                query = replace_cur_year(query)
                cursor = get_cursor_from_path(sqlite_path)
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    cursor.close()
                    cursor.connection.close()
                    return "result", result
                except Exception as e:
                    cursor.close()
                    cursor.connection.close()
                    return "exception", e
            
            flag, _ = exec_on_db_(db_path, sql)
            if flag == "exception":
                return 0
            else:
                return 1
        
        for attempt in range(max_retries):
            sql = self.generate_sql(input_sequence, max_retries=1)
            
            if sql and is_valid(sql, db_path):
                return sql
            else:
                print(f'SQL generation attempt {attempt + 1} failed validation: {sql}')
                if attempt < max_retries - 1:
                    continue
        return sql

    def generate_with_fallback_and_validation(self,
                                             db_schema: str,
                                             input_sequence: str,
                                             original_question: str,
                                             db_path: str) -> Optional[str]:
        """
        先用改写后的问题生成SQL，最多重试3次；失败则用原始问题再试最多3次；
        最终返回最后一次生成的SQL（可能未通过验证）。
        """
        def build_input(schema: str, question: str) -> str:
            return f"""You are now a sqlite data analyst, and you are given a database schema as follows:

【Schema】
{schema}

【Question】
{question}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question with no explanations. The generated SQL is protected by ```sql and ```.
"""
        # 第一阶段：使用改写后的问题
        input_seq_rewrite = input_sequence
        last_sql = None
        for attempt in range(3):
            sql = self.generate_with_validation(input_seq_rewrite, db_path=db_path, max_retries=1)
            last_sql = sql
            if sql:
                return sql
        # 第二阶段：使用原始问题
        input_seq_original = build_input(db_schema, original_question)
        for attempt in range(3):
            sql = self.generate_with_validation(input_seq_original, db_path=db_path, max_retries=1)
            last_sql = sql
            if sql:
                return sql
        return last_sql


def create_sql_generator(model_name: str = "deepseek-v3") -> SQLGenerator:
    """创建SQL生成器实例"""
    return SQLGenerator(model_name)


# 便捷函数
def quick_generate_sql(input_sequence: str, 
                      model_name: str = "deepseek-v3",
                      max_retries: int = 5) -> Optional[str]:
    """快速生成SQL"""
    generator = create_sql_generator(model_name)
    return generator.generate_sql(input_sequence, max_retries)


def quick_generate_sql_with_validation(input_sequence: str, 
                                      db_path: str,
                                      model_name: str = "deepseek-v3",
                                      max_retries: int = 5) -> Optional[str]:
    """快速生成并验证SQL"""
    generator = create_sql_generator(model_name)
    return generator.generate_with_validation(input_sequence, db_path, max_retries)
