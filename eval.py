import yaml
import pandas as pd
from datasets import load_dataset
from ragas.metrics import answer_similarity
from ragas.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load config

def load_config(config_path="eval-config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl_dataset(dataset_id, mapper, sys_col=None):
    ds = load_dataset(dataset_id, split="train")
    df = pd.DataFrame(ds)
    # Rename columns according to mapper
    df = df.rename(columns=mapper)
    # Only keep mapped columns
    cols = [mapper['question'], mapper['answer']]
    if sys_col and sys_col in df.columns:
        cols.append(sys_col)
    return df[cols]

def merge_datasets(cfg):
    dfs = {}
    # LLM generated (ground truth)
    if cfg['use_llm_generated']:
        dfs['llm'] = load_jsonl_dataset(cfg['llm_generated_dataset'], cfg['llm_mapper'])
    # Copilot
    if cfg['use_copilot']:
        dfs['copilot'] = load_jsonl_dataset(cfg['copilot_dataset'], cfg['copilot_mapper'])
    # Base
    if cfg['use_base']:
        dfs['base'] = load_jsonl_dataset(cfg['base_dataset'], cfg['base_mapper'], cfg.get('system_prompt_column'))
    # Fine-tuned
    if cfg['use_fine_tuned']:
        dfs['fine_tuned'] = load_jsonl_dataset(cfg['fine_tuned_dataset'], cfg['fine_tuned_mapper'], cfg.get('system_prompt_column'))
    # Merge on question
    df = dfs['llm'].rename(columns={cfg['llm_mapper']['answer']: 'answer_llm'})
    if 'copilot' in dfs:
        df = df.merge(dfs['copilot'].rename(columns={cfg['copilot_mapper']['answer']: 'answer_copilot'}), on=cfg['llm_mapper']['question'], how='left')
    if 'base' in dfs:
        df = df.merge(dfs['base'].rename(columns={cfg['base_mapper']['answer']: 'answer_base'}), on=cfg['llm_mapper']['question'], how='left')
    if 'fine_tuned' in dfs:
        df = df.merge(dfs['fine_tuned'].rename(columns={cfg['fine_tuned_mapper']['answer']: 'answer_fine_tuned'}), on=cfg['llm_mapper']['question'], how='left')
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
    emb = OpenAIEmbeddings(model=cfg['ragas']['embedding_model'])
    metrics = {}
    cost_report = {}
    for col in ['answer_copilot', 'answer_base', 'answer_fine_tuned']:
        if col in df.columns:
            scores = answer_similarity(
                df['answer_llm'].astype(str).tolist(),
                df[col].astype(str).tolist(),
                embeddings=emb
            )
            df[f'ragas_{col}'] = scores
            metrics[col] = sum(scores) / len(scores)
            cost_report[col] = calculate_ragas_cost(len(scores), cfg['ragas']['embedding_model'])
    return df, metrics, cost_report

def main():
    cfg = load_config()
    df = merge_datasets(cfg)
    df.to_csv(cfg['output_merged_file'], index=False)
    df, metrics, cost_report = run_ragas_eval(df, cfg)
    pd.DataFrame([metrics]).to_csv(cfg['output_metrics_file'], index=False)
    pd.DataFrame([cost_report]).to_csv("ragas_costs.csv", index=False)
    print("Ragas evaluation complete. Metrics:")
    print(pd.DataFrame([metrics]))
    print("Estimated Ragas cost (USD):")
    print(pd.DataFrame([cost_report]))

if __name__ == "__main__":
    main()
