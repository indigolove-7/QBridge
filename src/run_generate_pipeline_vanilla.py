import argparse
import json
import openai
import os
import re
import sqlite3
import random
import itertools

# 设置环境变量 - 必须在导入其他模块之前
# add your openai api key
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

from sql_post_process import fix_select_column
from get_selfconsistent_output import get_sqls
from tqdm import tqdm
import torch.nn.functional as F


from sqlalchemy import create_engine
from schema_engine_sqlite import SchemaEngine
from schema_linking import get_schema_linking_info
from question_rewrite import create_rewriter
from rewrite_reviewer import create_reviewer
from query_refiner import create_query_refiner
from sql_refiner import create_sql_refiner
from sql_generator import create_sql_generator
from reverse_deducer import create_reverse_deducer
from sql_execution_checker import create_execution_checker

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


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


mschema_prompt = """You are now a sqlite data analyst, and you are given a database schema as follows:

【Schema】
{db_schema}

【Question】
{question}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question with no explanations. The generated SQL is protected by ```sql and ```.
"""

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generate sqls")
    parser.add_argument("--input_dataset_path", default="../data/spider/dev.json",type=str)
    parser.add_argument("--self_consistent", type=bool, default=False)
    # parser.add_argument("--n", type=int, default=20,
    #                     help="Size of self-consistent set")
    parser.add_argument("--n", type=int, default=20,
                        help="Size of self-consistent set")
    parser.add_argument("--output_dataset_path", default="./pred.txt",type=str)
    parser.add_argument("--db_root", type=str, default="../data/spider/database")
    parser.add_argument("--rewrite_model_path", type=str, default="/home/lzs/models/ds_distilled_qwen_lora_temp0/sft_less_structured_ds_low_temp")
    parser.add_argument("--is_lora", type=bool, default=False, help="是否使用LoRA模型")
    parser.add_argument("--base_model_path", type=str, default="/home/lzs/models/ds_distilled_qwen_lora_temp0/sft_less_structured_ds_low_temp", help="LoRA模型的基础模型路径")
    parser.add_argument("--sl_sc_num", type=int, default=1, help="Size of schema linking self-consistent set")
    parser.add_argument("--use_baseline_schema", type=bool, default=False, help="使用baseline的schema linking结果，跳过自己的schema linking流程")
    parser.add_argument("--baseline_schema_path", type=str, default="./generate_datasets/C3_dev.json", help="baseline schema linking结果文件路径")
    # 统一配置各子模块/阶段使用的大模型
    parser.add_argument("--schema_linking_model", type=str, default="deepseek-v3", help="LLM for schema linking (tables/columns recall)")
    parser.add_argument("--review_model", type=str, default="deepseek-v3", help="LLM for rewrite reviewer")
    parser.add_argument("--query_refine_model", type=str, default="deepseek-v3", help="LLM for query refiner")
    parser.add_argument("--sql_refine_model", type=str, default="deepseek-v3", help="LLM for sql refiner")
    parser.add_argument("--sqlgen_model", type=str, default="deepseek-v3", help="LLM for final SQL generation")
    parser.add_argument("--self_consistency_model", type=str, default="deepseek-v3", help="LLM for self-consistency sampling when enabled")
    parser.add_argument("--reverse_deduce_model", type=str, default="deepseek-v3", help="LLM for reverse deduction from SQL to Gold Query Norm")
    parser.add_argument("--enable_reverse_deduct", type=bool, default=True, help="Enable reverse deduction after SQL generation")
    parser.add_argument("--sql_refine_feedback_model", type=str, default="deepseek-v3", help="LLM for generating SQL refine feedback from reverse query vs gold query")
    parser.add_argument("--execution_checker_model", type=str, default="deepseek-v3", help="LLM for SQL execution checking and feedback generation")
    parser.add_argument("--enable_execution_check", type=bool, default=True, help="Enable SQL execution checking for general feedback")
    parser.add_argument("--enable_sql_refine", type=bool, default=True, help="Enable SQL refinement based on execution and semantic feedback")
    parser.add_argument("--rewrite_device", type=str, default="cuda:1", help="Device for question rewriter (e.g., 'cuda:1', 'cuda:2', 'cpu')")
    opt = parser.parse_args()

    return opt


def normalize_question(question):
    """标准化问题字符串，处理Unicode单引号等特殊字符"""
    return question.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace("\u201d", "'").strip()

def load_baseline_schema_data(file_path):
    """加载baseline的schema linking结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建以(db_id, question)为键的字典，方便快速查找
    schema_dict = {}
    for item in data:
        # 标准化问题字符串，处理Unicode单引号等特殊字符
        normalized_question = normalize_question(item['question'])
        key = (item['db_id'], normalized_question)
        schema_dict[key] = item
    
    return schema_dict

opt = parse_option()
print(opt)

# 如果使用baseline schema，预先加载数据
baseline_schema_data = None
if opt.use_baseline_schema:
    print(f"正在加载baseline schema数据: {opt.baseline_schema_path}")
    baseline_schema_data = load_baseline_schema_data(opt.baseline_schema_path)
    print(f"已加载 {len(baseline_schema_data)} 条baseline schema记录")


query_rewriter = create_rewriter(opt.rewrite_model_path, opt.rewrite_device, opt.is_lora, opt.base_model_path)
query_reviewer = create_reviewer(model_name=opt.review_model)
query_refiner = create_query_refiner(model_name=opt.query_refine_model)
sql_generator = create_sql_generator(model_name=opt.sqlgen_model)
sql_refiner = create_sql_refiner(model_name=opt.sql_refine_model)
self_consistency_sql_generator = create_sql_generator(model_name=opt.self_consistency_model)

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


def is_valid(sql, db_path):
    flag, _ = exec_on_db_(db_path, sql)
    if flag == "exception":
        return 0
    else:
        return 1

if __name__ == '__main__':

    with open(opt.input_dataset_path) as f:
        data = json.load(f)

    all_db_infos = json.load(open("../data/spider/tables.json"))
    # 预构建 db_id -> db_info 映射，加速查找
    db_info_by_id = { db["db_id"]: db for db in all_db_infos }
    
    results = []
    p_sql_final = []

    all_questions = []
    all_first_sqls = []

    if not opt.self_consistent:
        for i, item in enumerate(data):
            print("id", i)

            db_name = item['db_id']
            question = item['question']
            print(f"db_name: {db_name}")
            # 获取数据库的完整schema信息
            db_info = db_info_by_id[db_name]
            db_path = f"{opt.db_root}/{db_name}/{db_name}.sqlite"
            print(db_path)
            assert os.path.exists(db_path)
            abs_path = os.path.abspath(db_path)
            
            # 根据配置选择使用baseline schema或自己的schema linking
            if opt.use_baseline_schema:
                # 使用baseline的schema linking结果
                normalized_question = normalize_question(question)
                key = (db_name, normalized_question)
                if key in baseline_schema_data:
                    baseline_item = baseline_schema_data[key]
                    # 从baseline数据中提取表名（从schema字段）
                    selected_tables = list(baseline_item['schema'].keys())
                    # 从input_sequence中提取列信息（这里简化处理，实际可能需要更复杂的解析）
                    selected_columns = []
                    for table, columns in baseline_item['schema'].items():
                        for col in columns:
                            selected_columns.append(f"{table}.{col}")
                else:
                    print(f"警告: 在baseline数据中未找到 {key}，回退到自己的schema linking")
                    # 直接报错终止程序
                    raise ValueError(f"在baseline数据中未找到 {key}")
            else:
                # 使用自己的schema linking
                schema_linking_info = get_schema_linking_info(question, db_name, abs_path, db_info, opt.sl_sc_num, model_name=opt.schema_linking_model)
                selected_tables = schema_linking_info['selected_tables']
                selected_columns = schema_linking_info['selected_columns']
            input_tables = [f"main.{table}" for table in selected_tables]
            input_columns = selected_columns

            # 生成纯净的schema用于问题重写和SQL生成
            db_engine = create_engine(f"sqlite:///{abs_path}")
            schema_engine = SchemaEngine(engine=db_engine, schema="main", db_name=db_name)
            mschema = schema_engine.mschema
            mschema_str = mschema.to_mschema(selected_tables=input_tables, selected_columns=input_columns)

            # rewrite the question
            rewritten_query = query_rewriter.rewrite_question(question, mschema_str)
            print(f"question: {question}")
            print(f"rewritten_query: {rewritten_query}")
            print()

            # review the rewritten question
            rubber_duck_feedback = query_reviewer.evaluate_single(question, rewritten_query, mschema_str)  # 返回rubber_duck_feedback + fix_suggestions
            print(f"rubber_duck_feedback: {rubber_duck_feedback}")
            # refine the rewritten question
            fixes = None
            try:
                fixes = rubber_duck_feedback.get("fixes", None) if isinstance(rubber_duck_feedback, dict) else None
            except Exception:
                fixes = None

            if isinstance(fixes, list) and len(fixes) == 0:
                gold_query = rewritten_query
            else:
                # 仅将 fixes 传入 refiner，避免无关信息干扰
                reviewer_feedback_for_refine = {"fixes": fixes} if isinstance(fixes, list) else rubber_duck_feedback
                gold_query = query_refiner.refine(question, rewritten_query, mschema_str, reviewer_feedback_for_refine)
            print(f"gold_query: {gold_query}")



            # 使用封装的SQL生成器
            p_sql = sql_generator.generate_with_validation(
                input_sequence=mschema_prompt.format(db_schema=mschema_str, question=gold_query),
                db_path=abs_path,
                max_retries=5
            )
            print(f"p_sql: {p_sql}")
            print("================================================")

            # 可选：对生成的SQL进行逆向推导，并基于 gold_query 生成 SQL refine 反馈
            semantic_consistency_feedback = None
            if opt.enable_sql_refine and opt.enable_reverse_deduct:
                reverse_query = None
                feedback_json = None
                try:
                    db_schema_for_reverse = mschema_str
                    reverse_deductor = create_reverse_deducer(model_name=opt.reverse_deduce_model)
                    # 单次调用：同时拿到 reverse query 和 feedback
                    combined = reverse_deductor.deduce_query_and_feedback(
                        sql=p_sql,
                        question=gold_query,
                        db_schema=db_schema_for_reverse,
                    )
                    if combined:
                        reverse_query = combined.get("reverse_query")
                        semantic_consistency_feedback = combined.get("feedback")
                        print(f"reverse_query: {reverse_query}")
                        print(f"semantic_consistency_feedback: {json.dumps(semantic_consistency_feedback, ensure_ascii=False)}")
                        print("================================================")
                        
                        # 若语义一致，则不提供语义反馈（置为 None）
                        if semantic_consistency_feedback is not None:
                            flag = semantic_consistency_feedback.get("consistency_with_given_question")
                            if isinstance(flag, str):
                                flag_norm = flag.strip().lower() in ("true", "yes", "1")
                            else:
                                flag_norm = bool(flag)
                            if flag_norm:
                                semantic_consistency_feedback = None
                except Exception as e:
                    print(f"reverse deduct (combined) error: {e}")

            # 执行检查（若启用），用于生成执行层面反馈
            general_execution_feedback = None
            execution_checker = None
            if opt.enable_sql_refine and opt.enable_execution_check:
                execution_checker = create_execution_checker(model_name=opt.execution_checker_model)
                general_execution_feedback = execution_checker.check_sql_execution(
                    p_sql, abs_path, mschema_str, question
                )
                if general_execution_feedback:
                    print(f"Execution error: {general_execution_feedback}")

            # 语义/执行反馈驱动的多轮 refine
            if opt.enable_sql_refine and (semantic_consistency_feedback is not None or general_execution_feedback is not None):
                max_refine_rounds = 5
                for j in range(max_refine_rounds):
                    refined_sql = sql_refiner.refine(
                        original_question=question,
                        current_sql=p_sql,
                        db_schema=mschema_str,
                        semantic_feedback=semantic_consistency_feedback,
                        execution_feedback=general_execution_feedback
                    )
                    if not refined_sql:
                        print("SQL refine failed, keeping original SQL")
                        break

                    # 清理与标准化
                    refined_sql = sql_generator._post_process_sql(refined_sql)
                    print(f'refined_sql: {refined_sql}')

                    # 可执行性校验；若失败并启用执行检查，则更新执行反馈继续下一轮
                    if is_valid(refined_sql, abs_path):
                        break
                    else:
                        print(f'refine_id: {j} p_sql: {refined_sql} exec error...')
                        if j < max_refine_rounds - 1:
                            print('refine again')

            p_sql_final.append(p_sql)
            # print(p_sql_final)
    else:
        for i, item in enumerate(tqdm(data)):
        # 只遍历前10个数据
        # for i in range(10):
            item = data[i]
            db_name = item['db_id']
            question = item['question']
            db_info = db_info_by_id[db_name]
            db_path = f"{opt.db_root}/{db_name}/{db_name}.sqlite"
            assert os.path.exists(db_path)
            abs_path = os.path.abspath(db_path)
            p_sqls = []
            db_engine = create_engine(f"sqlite:///{abs_path}")
            schema_engine = SchemaEngine(engine=db_engine, schema="main", db_name=db_name)
            mschema = schema_engine.mschema
            mschema_str = mschema.to_mschema()
            
            rewritten_query = query_rewriter.rewritten_query(question, mschema_str)
            # 使用封装的SQL生成器进行self-consistency
            
            for j in range(5):
                try:
                    # 生成多个SQL用于self-consistency
                    p_sqls = self_consistency_sql_generator.generate_multiple_sqls(
                        input_sequence=mschema_prompt.format(db_schema=mschema_str, question=rewritten_query),
                        n=opt.n,
                        use_additional_tips=True
                        # 使用默认温度1.0，与原始代码保持一致
                    )
                    
                    if p_sqls and is_valid(p_sqls[0], abs_path):
                        break
                    else:
                        print(f're_id: {j} p_sql: {p_sqls[0] if p_sqls else "None"} exec error...')
                        if j < 4:
                            print(f'generate again')
                            
                except Exception as e:
                    print(f"SQL generation error: {e}")
                    print(f"api error, retry...")

            print("")
            result = {}
            result['db_id'] = item['db_id']
            result['question'] = item['question']
            result['p_sqls'] = []
            for sql in p_sqls:
                result['p_sqls'].append(sql)
            results.append(result)
            
        p_sql_final = get_sqls(results, opt.n, opt.db_root)

    
    with open(opt.output_dataset_path, 'w') as f:
        for sql in p_sql_final:
            print(sql, file=f)
