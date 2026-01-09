from utils import examples_to_str, read_json, write_json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class MSchema:
    def __init__(self, db_id: str = 'Anonymous', schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []

    def add_table(self, name, fields={}, comment=None):
        self.tables[name] = {"fields": fields.copy(), 'examples': [], 'comment': comment}

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
            primary_key: bool = False, nullable: bool = True, default: Any = None,
            autoincrement: bool = False, comment: str = "", examples: list = [], **kwargs):
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f'{default}',
            "autoincrement": autoincrement,
            "comment": comment,
            "examples": examples.copy(),
            **kwargs}

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type, simple_mode=True)->str:
        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def has_table(self, table_name: str) -> bool:
        if table_name in self.tables.keys():
            return True
        else:
            return False

    def has_column(self, table_name: str, field_name: str) -> bool:
        if self.has_table(table_name):
            if field_name in self.tables[table_name]["fields"].keys():
                return True
            else:
                return False
        else:
            return False

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        try:
            return self.tables[table_name]['fields'][field_name]
        except:
            return {}

    def single_table_mschema(self, table_name: str, selected_columns: List = None,
                             example_num=3, show_type_detail=False) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get('comment', '')
        if table_comment is not None and table_comment != 'None' and len(table_comment) > 0:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema is not None and len(self.schema) > 0:
                # output.append(f"# Table: {self.schema}.{table_name}")
                output.append(f"# Table: {table_name.split('.')[1]}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        # 处理表中的每一个字段
        for field_name, field_info in table_info['fields'].items():
            if selected_columns is not None and field_name.lower() not in selected_columns:
                continue

            raw_type = self.get_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"
            if field_info['comment'] != '':
                field_line += f", {field_info['comment'].strip()}"
            else:
                pass

            ## 打上主键标识
            is_primary_key = field_info.get('primary_key', False)
            if is_primary_key:
                field_line += f", Primary Key"

            # 如果有示例，添加上
            if len(field_info.get('examples', [])) > 0 and example_num > 0:
                examples = field_info['examples']
                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]]
                elif len(examples) > 0 and max([len(s) for s in examples]) > 20:
                    if max([len(s) for s in examples]) > 50:
                        examples = []
                    else:
                        examples = [examples[0]]
                else:
                    pass
                if len(examples) > 0:
                    example_str = ', '.join([str(example) for example in examples])
                    field_line += f", Examples: [{example_str}]"
                else:
                    pass
            else:
                field_line += ""
            field_line += ")"

            field_lines.append(field_line)
        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')

        return '\n'.join(output)

    def to_mschema(self, selected_tables: List = None, selected_columns: List = None,
                   example_num=3, show_type_detail=False) -> str:
        """
        convert to a MSchema string.
        selected_tables: 默认为None，表示选择所有的表
        selected_columns: 默认为None，表示所有列全选，格式['table_name.column_name']
        """
        output = []

        # output.append(f"【DB_ID】 {self.db_id}")
        # output.append(f"【Schema】")

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]
        if selected_columns is not None:
            selected_columns = [s.lower() for s in selected_columns]
            # selected_tables = [s.split('.')[0].lower() for s in selected_columns]

        # 首先收集所有需要显示的外键列
        fk_columns = set()
        if self.foreign_keys and selected_tables is not None:
            for fk in self.foreign_keys:
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                T1 = ("main."+table1.replace(self.schema+".",""))
                T2 = ("main."+table2.replace(self.schema+".",""))
                if (T1.lower() in selected_tables and T2.lower() in selected_tables and ref_schema == self.schema):
                    # 添加外键列到需要显示的列集合中
                    table1_name = T1.replace("main.", "").lower()
                    table2_name = T2.replace("main.", "").lower()
                    fk_columns.add(f"{table1_name}.{column1.lower()}")
                    fk_columns.add(f"{table2_name}.{column2.lower()}")

        # 依次处理每一个表
        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                cur_table_type = table_info.get('type', 'table')
                column_names = list(table_info['fields'].keys())
                if selected_columns is not None:
                    # 包含原始选中的列和外键列
                    cur_selected_columns = [c.lower() for c in column_names if f"{table_name.split('.')[1]}.{c}".lower() in selected_columns]
                    # 添加外键列
                    table_name_short = table_name.split('.')[1].lower()
                    for fk_col in fk_columns:
                        if fk_col.startswith(f"{table_name_short}."):
                            col_name = fk_col.split('.')[1]
                            if col_name not in cur_selected_columns:
                                cur_selected_columns.append(col_name)
                else:
                    cur_selected_columns = selected_columns
                output.append(self.single_table_mschema(table_name, cur_selected_columns, example_num, show_type_detail))

        # 添加外键信息，选择table_type为view时不展示外键
        if self.foreign_keys:
            # 先收集所有有效的外键
            valid_fks = []
            for fk in self.foreign_keys:
                # print(fk)
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                T1 = ("main."+table1.replace(self.schema+".",""))
                T2 = ("main."+table2.replace(self.schema+".",""))
                if selected_tables is None or \
                        (T1.lower() in selected_tables and T2.lower() in selected_tables):
                    if ref_schema == self.schema:
                        # 修改：只要两端的表都被选中，就添加外键信息（与schema_linking.py保持一致）
                        valid_fks.append(f"{T1.replace('main.','')}.{fk[1]}={T2.replace('main.','')}.{fk[4]}")
            
            # 只有当有有效外键时才添加标题和外键信息
            if valid_fks:
                output.append("【Foreign keys】")
                output.extend(valid_fks)

        return '\n'.join(output)

    def dump(self):
        schema_dict = {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys
        }
        return schema_dict

    def save(self, file_path: str):
        schema_dict = self.dump()
        write_json(file_path, schema_dict)

    def load(self, file_path: str):
        data = read_json(file_path)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])
