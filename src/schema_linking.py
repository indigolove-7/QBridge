"""
Schema Linking模块
包含表召回、列召回、外键处理等核心功能
"""

import json
import time
import openai
import os
from sqlalchemy import create_engine
from schema_engine_sqlite import SchemaEngine
from table_recall import table_sc
from column_recall import column_sc

# OpenAI客户端配置
# 使用环境变量配置 OpenAI 客户端，避免硬编码
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 表召回指令
table_recall_instruction = """Given the database schema and question, perform the following actions: 
1 - Rank all the tables based on the possibility of being used in the SQL according to the question from the most relevant to the least relevant, Table or its column that matches more with the question words is highly relevant and must be placed ahead.
2 - Check whether you consider all the tables.
3 - Output a list object in the order of step 2, Your output should contain all the tables. The format should be like: 
[
    "table_1", "table_2", ...
]

【Schema】
{schema}

【Question】
{question}
"""

# 列召回指令
column_recall_instruction = """Given the database schema and question, perform the following actions: 
1 - Rank the columns in each table based on the possibility of being used in the SQL, Column that matches more with the question words or the foreign key is highly relevant and must be placed ahead. You should output them in the order of the most relevant to the least relevant.
2 - Output a JSON object that contains all the columns in each table according to your explanation. The format should be like: 
{{
    "table_1": ["column_1", "column_2", ......], 
    "table_2": ["column_1", "column_2", ......],
    "table_3": ["column_1", "column_2", ......],
     ......
}}

【Schema】
{schema}

【Question】
{question}

"""


def generate_reply_for_table_recall(input, sc_num, model_name="deepseek-v3"):
    """生成表召回的回复"""
    completions = client.chat.completions.create(
        model=model_name,
        messages=input,
        temperature=0.0,
        n=sc_num
    )
    all_tables = []
    for i in range(sc_num):
        raw_table = completions.choices[i].message.content
        try:
            raw_table = '[' + raw_table.split('[', 1)[1]
            raw_table = raw_table.rsplit(']', 1)[0] + ']'
            raw_table = eval(raw_table)
            if Ellipsis in raw_table:
                raw_table.remove(Ellipsis)
        except:
            print('list error')
            return None
        all_tables.append(raw_table)
    return all_tables


def generate_reply_for_column_recall(input, sc_num, model_name="deepseek-v3"):
    """生成列召回的回复"""
    completions = client.chat.completions.create(
        model=model_name,
        messages=input,
        temperature=0.0,
        n=sc_num
    )
    tabs_cols_all = []
    for i in range(sc_num):
        raw_tab_col = completions.choices[i].message.content
        try:
            raw_tab_col = '{' + raw_tab_col.split('{', 1)[1]
            raw_tab_col = raw_tab_col.rsplit('}', 1)[0] + '}'
            raw_tab_col = json.loads(raw_tab_col)
        except:
            print('list error')
            return None
        tabs_cols_all.append(raw_tab_col)
    return tabs_cols_all


def perform_table_recall(question, mschema_str, db_info, sc_num, model_name="deepseek-v3"):
    """执行表召回步骤"""
    table_recall_mschema_str = mschema_str.split("【Foreign keys】")[0]
    prompt = table_recall_instruction.format(schema=table_recall_mschema_str, question=question)
    tables_all = None
    retry_count = 0
    max_retries = 5
    
    while tables_all is None and retry_count < max_retries:
        try:
            tables_all = generate_reply_for_table_recall([{"role": "user", "content": prompt}], sc_num, model_name=model_name)
            if tables_all and len(tables_all) > 0:
                # 验证返回格式 - 应该是列表的列表
                if not isinstance(tables_all, list) or not all(isinstance(item, list) for item in tables_all):
                    tables_all = None
                    raise ValueError("Invalid table list format")
            else:
                raise ValueError("Empty response from table recall API")
        except Exception as e:
            print(f'Table recall API error: {e}, retry... (attempt {retry_count + 1}/{max_retries})')
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached for table recall, using empty list")
                tables_all = []
    
    # 构建原始表名列表
    tables_ori = [table.lower() for table in db_info["table_names_original"]]
    
    # 使用table_sc进行自一致性处理
    tables = table_sc(tables_all, tables_ori)
    return tables, tables_all


def perform_column_recall(question, mschema_str, db_info, sc_num, model_name="deepseek-v3"):
    """执行列召回步骤"""
    column_recall_mschema_str = mschema_str
    prompt = column_recall_instruction.format(schema=column_recall_mschema_str, question=question)
    columns_all = None
    retry_count = 0
    max_retries = 5
    
    while columns_all is None and retry_count < max_retries:
        try:
            columns_all = generate_reply_for_column_recall([{"role": "user", "content": prompt}], sc_num, model_name=model_name)
            if columns_all and len(columns_all) > 0:
                # 验证返回格式 - 应该是字典的列表
                if not isinstance(columns_all, list) or not all(isinstance(item, dict) for item in columns_all):
                    columns_all = None
                    raise ValueError("Invalid column dict format")
            else:
                raise ValueError("Empty response from column recall API")
        except Exception as e:
            print(f'Column recall API error: {e}, retry... (attempt {retry_count + 1}/{max_retries})')
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached for column recall, using empty dict")
                columns_all = {}
    
    # 构建表到列的映射
    tab_col_ori = {}
    for table_idx, table_name in enumerate(db_info["table_names_original"]):
        table_columns = []
        for col_idx, (col_table_idx, col_name) in enumerate(db_info["column_names_original"]):
            if col_table_idx == table_idx:
                table_columns.append(col_name)
        tab_col_ori[table_name.lower()] = table_columns
    
    return columns_all, tab_col_ori


def process_foreign_keys(db_info, selected_tables):
    """处理外键关系"""
    foreign_keys = []
    table_names_original = db_info["table_names_original"]
    column_names_original = db_info["column_names_original"]
    
    # 构建外键列表
    for source_column_idx, target_column_idx in db_info["foreign_keys"]:
        fk_source_table_name_original = table_names_original[column_names_original[source_column_idx][0]]
        fk_source_column_name_original = column_names_original[source_column_idx][1]
        fk_target_table_name_original = table_names_original[column_names_original[target_column_idx][0]]
        fk_target_column_name_original = column_names_original[target_column_idx][1]

        foreign_keys.append({
            "source_table_name_original": fk_source_table_name_original.lower(),
            "source_column_name_original": fk_source_column_name_original.lower(),
            "target_table_name_original": fk_target_table_name_original.lower(),
            "target_column_name_original": fk_target_column_name_original.lower(),
        })
    
    # 过滤出相关的外键
    db_info['fk'] = []
    for fk in foreign_keys:
        if (fk['source_table_name_original'] in selected_tables and 
            fk['target_table_name_original'] in selected_tables):
            fk_str = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
                    f"{fk['target_table_name_original']}.{fk['target_column_name_original']}"
            db_info['fk'].append(fk_str)
    
    return db_info['fk']


def get_schema_linking_info(question, db_name, abs_path, db_info, sc_num, model_name="deepseek-v3"):
    """主要的schema linking函数，协调表召回和列召回"""
    # 初始化数据库连接和schema
    db_engine = create_engine(f"sqlite:///{abs_path}")
    schema_engine = SchemaEngine(engine=db_engine, schema="main", db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()

    # 步骤1: 表召回
    selected_tables, tables_all = perform_table_recall(question, mschema_str, db_info, sc_num, model_name=model_name)
    print(selected_tables)
    # 步骤2: 列召回
    columns_all, tab_col_ori = perform_column_recall(question, mschema_str, db_info, sc_num, model_name=model_name)
    print(columns_all)
    # 步骤3: 处理外键
    foreign_keys = process_foreign_keys(db_info, selected_tables)
    print(foreign_keys)
    # 步骤4: 列自一致性处理
    columns = column_sc(columns_all, tab_col_ori, foreign_keys)

    # 步骤5: 构建返回结果
    schema_linking_info = {
        'selected_tables': selected_tables,
        'selected_columns': [f"{table}.{col}" for table, cols in columns.items() for col in cols],
        'unselected_tables': [table for table in db_info["table_names_original"] 
                             if table.lower() not in selected_tables],
        'unselected_columns': [f"{table}.{col}" for table in db_info["table_names_original"] 
                              if table.lower() not in selected_tables 
                              for col in tab_col_ori.get(table.lower(), [])],
    }
    return schema_linking_info
