# QBridge: Bridging Natural Language and SQL via Gold Query Rewriting with Agentic Refinement


## Paper
This repository accompanies the paper:
- **QBridge: Bridging Natural Language and SQL via Gold Query Rewriting with Agentic Refinement**

## Directory Structure
```
QBridge_src/
 ├── README.md          
 ├── my_env.yaml         # Conda environment file
 ├── dataset/            # Distilled Back-Translation data for SL-Independent NLQ Rewriter training
 │    └── data_ds_5000.json
 ├── data/               # Spider and benchmark datasets for evaluation
 │   ├── spider/        
 │   ├── spider-dk/      
 │   ├── spider-realistic/ 
 │   └── spider-syn/     
 └── src/                # Core framework source code
      ├── question_rewrite.py     
      ├── rewrite_reviewer.py     
      ├── query_refiner.py        
      ├── reverse_deducer.py      
      ├── run_generate_pipeline_vanilla.py  # Main pipeline script
      ├── ...                     # Other modules: SQL post-process, execution checker, etc.
```

- `dataset/`: Contains data created under the paper's Distilled Back-Translation process for training the SL-independent NLQ Rewriter (`data_ds_5000.json`).
- `data/`: Contains full Spider benchmark suite (and variants), with Spider dev/train splits as well as realistic/distilled/synthetic sets.

## Environment
Install all dependencies using Conda:
```bash
conda env create -n qbridge -f my_env.yaml
conda activate qbridge
```
`my_env.yaml` includes core dependencies: PyTorch, Transformers, OpenAI, LLaMA-Factory, etc.

## Quick Start
1. Download/prepare the datasets in `data/` and `dataset/` (see Spider and DBT dataset descriptions).
      - Spider databases: https://github.com/taoyds/spider
      - TS databases: https://github.com/taoyds/test-suite-sql-eval
2. Configure your OpenAI API KEY (or use compatible LLM providers).
3. Run main pipeline:
```bash
python src/run_generate_pipeline_vanilla.py --input_dataset_path ./data/spider/dev.json --output_dataset_path ./output_sql.txt
```
See script comments and argparse help for detailed parameterization and modular entry-points (NLQ rewriting, reviewing, SQL generation, etc.).

## SL-Independent NLQ Rewriter Model

We provide several variants of our SL-Independent NLQ Rewriter models on [ModelScope (jaysonl)](https://modelscope.cn/profile/jaysonl). 
All versions can be downloaded from our ModelScope profile:
[https://modelscope.cn/profile/jaysonl](https://modelscope.cn/profile/jaysonl)

### Download and Setup
1. **Install ModelScope:**
```bash
pip install modelscope
```

2. **Download your chosen NLQ Rewriter model:**
Replace `<model_id>` with the desired version (refer to the ModelScope page for available model ids):
```bash
modelscope download --model <model_id> --local_dir /path/to/your/model
```
For example, for the full parameter fine-tuned version:
```bash
modelscope download --model jaysonl/ds_distilled_qwen_rewriter_sft_full --local_dir /path/to/your/model
```

3. **Load in Your Code:**
Specify the local directory in your model loader, e.g. `create_rewriter(model_path='/path/to/your/model', is_lora=True/False)` depending on model type.

> **Note:** Inference with these models typically requires at least 15GB of GPU VRAM (depending on the base model size and version).

---

## Core Modules
- **question_rewrite.py:** Schema-aware linguistic rewriter (NL → Gold Query); supports Qwen2.5-7B Instruct and LoRA adapter fine-tuning.
- **rewrite_reviewer.py:** Multi-dimensional structured feedback and advice via rubber duck evaluation.
- **query_refiner.py:** Gold Query refinement with rubber duck review and actionable fixes.
- **reverse_deducer.py:** Reverse Query deduction from SQL for analysis and error tracing.
- **run_generate_pipeline_vanilla.py:** End-to-end pipeline orchestrating the above phases with full modularity.

## Dataset Description
- `dataset/data_ds_5000.json`: The auto-generated (NL, Gold Query) pairs via Distilled Back-Translation as described in the paper (for training the NLQ Rewriter).
- `data/spider/`: Official Spider dataset (see `README.txt` inside for citation and format), EMNLP 2018.
- `data/spider-dk`, `spider-syn`, `spider-realistic`: Distilled/translated, synthetic, and real-world distribution variants for further evaluation.

## Useful Links
- Spider Dataset: https://yale-lily.github.io/spider
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- (Add more links once open-sourced)


