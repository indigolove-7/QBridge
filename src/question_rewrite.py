"""
问题重写模块
包含问题重写相关的功能和配置
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 重写相关的提示词配置
rewrite_system_prompt = """You are now an professional question rewriting specialist. I will provide a natural language question, the corresponding database schema. Your task is to rewrite the question into a semantically equivalent, structured, yet natural Gold Query format."""

# rewrite_instruction_prompt = """
# CRITICAL RULES:
# 1. SEMANTIC EQUIVALENCE: The rewritten question must preserve the exact meaning and attribute order of the original question
# 2. STRUCTURED FORMAT: Follow these structure patterns based on query type:
#    Base pattern: "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]"
#    UNION pattern: (for "either...or" logic):
#    - Keywords: "either...or", "or...either", "one of", "any of"
#    - Format: "List [attributes] from [table(s)] where [condition1] UNION [attributes] from [table(s)] where [condition2]"
#    INTERSECT pattern: (for "both...and" logic):
#    - Keywords: "both...and", "who both", "that both", "when both", "in which...both", "speak both", "voted both"
#    - Format: "List [attributes] from [table(s)] where [condition1] INTERSECT [attributes] from [table(s)] where [condition2]"
#    EXCEPT pattern: (for "but not" logic):
#    - Keywords: "but do not", "who do not", "that are not", "that does not", "not used by", "not have any", "not own any"
#    - Format: "List [attributes] from [table(s)] where [condition1] EXCEPT [attributes] from [table(s)] where [condition2]"
# 3. CONCISENESS: Remove all redundant words while maintaining clarity
# 4. UNAMBIGUITY: Explicitly specify aggregations (distinct, count, sum, avg, max, min) and comparisons
# 5. NATURAL LANGUAGE: Maintain natural English flow while using precise database terminology
# 6. OUTPUT ONLY: Return only the rewritten question, no explanations
# """
# """

rewrite_instruction_prompt_gpt4o = """
CRITICAL RULES:
1. SEMANTIC EQUIVALENCE: The rewritten question must preserve the exact meaning and attribute order of the original question
2. STRUCTURED FORMAT: Follow this exact structure but adapt it to the query's complexity::
   - Base pattern: "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]"
   - For subqueries: Enclose the subquery description in parentheses () to enhance hierarchical structure, e.g., "where the value is in the result of (subquery description)"
   - For join operations: Enclose each pair of join operations in parentheses () to enhance structural clarity, e.g., "from ((table1 join table2 on condition) join table3 on condition)"
   - For conditions: Express them as naturally as possible, not as SQL fragments, e.g, use "where the value is greater than 30" instead of "where value > 30".
3. CONCISENESS: Remove all redundant words while maintaining clarity
4. UNAMBIGUOUS: Explicitly specify aggregations (count, sum, avg, max, min) and comparisons, and distinguish between column names and aggregation functions
5. NATURAL LANGUAGE: Maintain natural English flow while using precise database terminology
6. OUTPUT ONLY: Return only the rewritten question with no explanations
"""

rewrite_instruction_prompt = """
CRITICAL RULES:
1. SEMANTIC EQUIVALENCE: The rewritten question must preserve the exact meaning and attribute order of the original question
2. STRUCTURED FORMAT: Follow this exact structure:
   - "List [aggregation functions] [attributes] from [table(s)] [with join conditions] [with conditions] [grouped by] [ordered by] [limited to]"
3. CONCISENESS: Remove all redundant words while maintaining clarity
4. UNAMBIGUOUS: Explicitly specify aggregations (count, sum, avg, max, min) and comparisons
5. NATURAL LANGUAGE: Maintain natural English flow while using precise database terminology
6. OUTPUT ONLY: Return only the rewritten question, no explanations
"""
rewrite_input_prompt = """
【Schema】
{db_schema}

【Question】
{question}
"""


class QuestionRewriter:
    """问题重写器类"""
    
    def __init__(self, model_path, device=None, is_lora=False, base_model_path=None):
        """初始化重写器
        
        Args:
            model_path: 模型路径（对于LoRA模型，这是LoRA适配器路径）
            device: 设备
            is_lora: 是否为LoRA模型
            base_model_path: 基础模型路径（仅当is_lora=True时需要）
        """
        import torch
        
        # 检查是否已经加载过相同的模型，避免重复加载
        if not hasattr(QuestionRewriter, '_loaded_models'):
            QuestionRewriter._loaded_models = {}
        
        # 使用模型路径和设备作为缓存键
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        model_key = f"{model_path}_{device}_{is_lora}"
        if model_key not in QuestionRewriter._loaded_models:
            print(f"正在加载模型: {model_path} 到设备: {device}")
            
            if is_lora:
                # 加载LoRA模型
                if base_model_path is None:
                    # 从adapter_config.json中读取基础模型路径
                    import json
                    with open(f"{model_path}/adapter_config.json", "r") as f:
                        config = json.load(f)
                        base_model_path = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
                
                print(f"加载基础模型: {base_model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype="auto",
                    device_map={"": device},
                    trust_remote_code=True,
                )
                
                print(f"加载LoRA适配器: {model_path}")
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            else:
                # 加载普通模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map={"": device},
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 缓存模型和tokenizer
            QuestionRewriter._loaded_models[model_key] = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'device': device
            }
        else:
            print(f"复用已加载的模型: {model_path} 在设备: {device}")
            cached = QuestionRewriter._loaded_models[model_key]
            self.model = cached['model']
            self.tokenizer = cached['tokenizer']
    
    def generate_rewrite_reply(self, messages):
        """生成重写回复"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
        return response
    
    def rewrite_question(self, question, db_schema):
        """重写问题"""
        rewrite_prompt = rewrite_instruction_prompt + "\n" + rewrite_input_prompt.format(
            db_schema=db_schema, 
            question=question
        )
        rewrite_messages = [
            {"role": "system", "content": rewrite_system_prompt},
            {"role": "user", "content": rewrite_prompt}
        ]
        
        rewritten_question = self.generate_rewrite_reply(rewrite_messages)
        return rewritten_question


def create_rewriter(model_path, device=None, is_lora=False, base_model_path=None):
    """创建重写器实例的工厂函数
    
    Args:
        model_path: 模型路径
        device: 设备
        is_lora: 是否为LoRA模型
        base_model_path: 基础模型路径（仅当is_lora=True时需要）
    """
    return QuestionRewriter(model_path, device, is_lora, base_model_path)
