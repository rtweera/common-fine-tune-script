import yaml
import pandas as pd
from datasets import load_dataset
from ragas.metrics import answer_similarity as answer_similarity_metric
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load config

def load_config(config_path="eval-config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl_dataset(dataset_id, mapper, sys_col=None, data_file=None, suffix=None):
    # If data_file is provided, use the 'json' loader and point to the file in the repo
    if data_file:
        ds = load_dataset("json", data_files=f"hf://datasets/{dataset_id}/{data_file}")['train']
    else:
        ds = load_dataset(dataset_id, split="train")
    df = pd.DataFrame(ds)
    # Rename columns according to mapper
    answer_col = f'answer{suffix or ""}'
    df = df.rename(columns={mapper['question']: 'question', mapper['answer']: answer_col})
    # Only keep mapped columns
    cols = ['question', answer_col]
    if sys_col and sys_col in df.columns:
        cols.append(sys_col)
    return df[cols]

def merge_datasets(cfg):
    dfs = {}
    # LLM generated (ground truth)
    if cfg['use_llm_generated']:
        dfs['llm'] = load_jsonl_dataset(cfg['llm_generated_dataset'], cfg['llm_mapper'], data_file=cfg.get('llm_data_file'), suffix='_llm')
    # Copilot
    if cfg['use_copilot']:
        dfs['copilot'] = load_jsonl_dataset(cfg['copilot_dataset'], cfg['copilot_mapper'], data_file=cfg.get('copilot_data_file'), suffix='_copilot')
    # Base
    if cfg['use_base']:
        dfs['base'] = load_jsonl_dataset(cfg['base_dataset'], cfg['base_mapper'], cfg.get('system_prompt_column'), data_file=cfg.get('base_data_file'), suffix='_base')
    # Fine-tuned
    if cfg['use_fine_tuned']:
        dfs['fine_tuned'] = load_jsonl_dataset(cfg['fine_tuned_dataset'], cfg['fine_tuned_mapper'], cfg.get('system_prompt_column'), data_file=cfg.get('fine_tuned_data_file'), suffix='_fine_tuned')
    # Merge on standardized 'question'
    df = dfs['llm']
    if 'copilot' in dfs:
        df = df.merge(dfs['copilot'], on='question', how='left')
    if 'base' in dfs:
        df = df.merge(dfs['base'], on='question', how='left')
    if 'fine_tuned' in dfs:
        df = df.merge(dfs['fine_tuned'], on='question', how='left')
    return df

def calculate_ragas_cost(num_pairs, model_name):
    # For OpenAI text-embedding-3-small: $0.02 per 1M tokens (input+output)
    # Assume average 750 tokens per pair (adjust as needed)
    # Each answer_similarity call does 2 embeddings per pair (ground truth + candidate)
    # 1 token = ~4 chars, so estimate tokens from char count if needed
    # We'll just estimate cost by number of pairs for now
    price_per_million = 0.02
    tokens_per_pair = 750 * 2  # both answers
    total_tokens = num_pairs * tokens_per_pair
    cost = (total_tokens / 1_000_000) * price_per_million
    return cost

def run_ragas_eval(df, cfg):
    load_dotenv(override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    os.environ["OPENAI_API_KEY"] = openai_key
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=cfg['ragas']['embedding_model']))
    metrics = {}
    cost_report = {}
    from datasets import Dataset
    eval_results = {}
    for col in ['answer_copilot', 'answer_base', 'answer_fine_tuned']:
        if col in df.columns:
            eval_dataset = Dataset.from_pandas(
                df[['question', 'answer_llm', col]].rename(columns={
                    'question': 'question',
                    'answer_llm': 'ground_truth',
                    col: 'answer',
                })
            )
            results = evaluate(eval_dataset, metrics=[answer_similarity_metric], embeddings=emb)
            # Print available keys for debugging
            print(f"Available keys in results: {list(results._scores_dict.keys())}")
            # Try to get the first available key if 'answer_similarity' is not present
            key = 'semantic_similarity'
            if key not in results._scores_dict:
                key = list(results._scores_dict.keys())[0]
            # Use numpy to compute mean if results[key] is a list
            import numpy as np
            df[f'ragas_{col}'] = results[key]
            metrics[col] = np.mean(results[key])
            cost_report[col] = calculate_ragas_cost(len(results[key]), cfg['ragas']['embedding_model'])
    return df, metrics, cost_report

def main():
    cfg = load_config()
    df = merge_datasets(cfg)
    df.to_csv(cfg['output_merged_file'], index=False)
    df, metrics, cost_report = run_ragas_eval(df, cfg)
    # Save the merged dataframe with per-answer Ragas scores
    df.to_csv('merged_eval_data_with_ragas.csv', index=False)
    pd.DataFrame([metrics]).to_csv(cfg['output_metrics_file'], index=False)
    pd.DataFrame([cost_report]).to_csv("ragas_costs.csv", index=False)
    print("Ragas evaluation complete. Metrics:")
    print(pd.DataFrame([metrics]))
    print("Estimated Ragas cost (USD):")
    print(pd.DataFrame([cost_report]))

if __name__ == "__main__":
    main()
