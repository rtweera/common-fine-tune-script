import requests
import json
from dotenv import load_dotenv
import os
from datetime import datetime
import yaml
from datasets import load_dataset
from tqdm import tqdm

from choreo_config import API_URL, HEADERS

load_dotenv()

def ask_question(question):
    """
    Ask a question from the choreo copilot, get the answer as a full text.
    """
    payload = json.dumps(
        {
            "question": question,
            "version": "v2.0",
            "current_datetime": datetime.now().isoformat(),
            "perspective": "dev",
        }
    )

    # print("awaiting response")
    response = requests.request(
        "POST", API_URL, headers=HEADERS, data=payload, stream=True
    )

    if response.status_code == 401:
        print(f"request failed\nResponse:{response.text}")
        return

    # print(f"response arrived:\n{response}")

    full_answer = ""
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
                if data.get("type") == "STREAM":
                    full_answer += data.get("content", "")
            except Exception as e:
                print(f"Exception: {e}")
                continue
    return full_answer


def ask_all_questions_and_save():
    # Load main config
    with open("choreo_chat_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_id = config["dataset_id"]
    data_file = config["datafile"]
    # Load dataset from HuggingFace
    ds = load_dataset("json", data_files=f"hf://datasets/{dataset_id}/{data_file}")["train"]
    output_path = "copilot_answers.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="Asking questions", unit="question"):
            question = item["question"]
            answer = ask_question(question)
            fout.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n")
    print(f"Saved copilot answers to {output_path}")


def push_copilot_answers_to_hf():
    from huggingface_hub import HfApi
    with open("choreo_chat_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hf_user = config.get("hf_user_id") or os.getenv("HF_USER_ID")
    repo_id = config.get("copilot_repo_id", "copilot-answers-dataset")
    output_path = "copilot_answers.jsonl"
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=f"{hf_user}/{repo_id}", repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="copilot_answers.jsonl",
        repo_id=f"{hf_user}/{repo_id}",
        repo_type="dataset",
    )
    print(f"Pushed copilot answers to HuggingFace: {hf_user}/{repo_id}")


if __name__ == "__main__":
    ask_all_questions_and_save()
    push_copilot_answers_to_hf()



