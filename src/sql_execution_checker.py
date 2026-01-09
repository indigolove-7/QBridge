"""
SQL Execution Checker 模块
- 获取生成的SQL在实际数据库上的执行反馈
"""

import sqlite3
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import time
import os


class SQLExecutionChecker:
    """SQL执行检查器，获取SQL在实际数据库上的执行反馈"""
    
    def __init__(self, model_name: str = "deepseek-v3"):
        self.model_name = model_name
    
    def check_sql_execution(self, 
                           sql: str, 
                           db_path: str, 
                           db_schema: str,
                           original_question: str) -> Optional[str]:
        """
        检查SQL在实际数据库上的执行情况，返回general_execution_feedback
        
        Args:
            sql: 要检查的SQL语句
            db_path: 数据库文件路径
            db_schema: 数据库schema信息
            original_question: 原始问题
            
        Returns:
            - 当执行出错时: 返回错误信息字符串
            - 当执行成功时: 返回None（表示无需general_execution_feedback）
        """
        try:
            execution_result = self._execute_sql(sql, db_path)
            if execution_result.get("executable"):
                return None
            return execution_result.get("error_message") or "Unknown SQL execution error"
        except Exception as e:
            return f"Execution failed: {str(e)}"
    
    def _execute_sql(self, sql: str, db_path: str) -> Dict[str, Any]:
        """执行SQL并返回执行结果"""
        feedback = {
            "executable": False,
            "error_message": None,
            "result_count": 0,
            "execution_time": 0
        }
        
        cursor = None
        try:
            start_time = time.time()
            
            # 替换当前年份
            processed_sql = self._replace_cur_year(sql)
            
            # 执行SQL
            cursor = self._get_cursor_from_path(db_path)
            cursor.execute(processed_sql)
            result = cursor.fetchall()
            
            end_time = time.time()
            
            feedback["executable"] = True
            feedback["result_count"] = len(result)
            feedback["execution_time"] = end_time - start_time
            
        except Exception as e:
            feedback["error_message"] = str(e)
            # executable 保持 False（默认值）
        finally:
            if cursor:
                cursor.close()
                cursor.connection.close()
        
        return feedback
    
    
    # 辅助方法
    def _replace_cur_year(self, query: str) -> str:
        """替换当前年份"""
        return re.sub(
            r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
        )
    
    def _get_cursor_from_path(self, sqlite_path: str):
        """获取数据库游标"""
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"Database file not found: {sqlite_path}")
        
        connection = sqlite3.connect(sqlite_path)
        connection.text_factory = lambda b: b.decode(errors="ignore")
        return connection.cursor()


def create_execution_checker(model_name: str = "deepseek-v3") -> SQLExecutionChecker:
    """创建SQL执行检查器实例"""
    return SQLExecutionChecker(model_name)


# 便捷函数
def quick_execution_check(sql: str, 
                        db_path: str, 
                        db_schema: str,
                        original_question: str,
                        model_name: str = "deepseek-v3") -> Optional[str]:
    """快速执行检查"""
    checker = create_execution_checker(model_name)
    return checker.check_sql_execution(sql, db_path, db_schema, original_question)
