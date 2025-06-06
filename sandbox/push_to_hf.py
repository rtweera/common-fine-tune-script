from dotenv import load_dotenv
from huggingface_hub import HfApi
import os


def push_to_hf(token=None):
    """
    Push a Dataset to Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        model_path (str): The local path to the model directory.
        token (str, optional): Hugging Face API token. If None, it will use the token from environment variables.
    """
    if token is None:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
    
    api = HfApi(token=token)
    api.upload_folder(
        folder_path='./output',
        repo_id="rtweera/output_2025-Jun-05_14-44-03_Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit_LoRA",
        repo_type="dataset",
        commit_message="Upload files",
    )
    print(f"Dataset pushed to successfully.")

if __name__ == "__main__":
    push_to_hf()
    print("Push to Hugging Face Hub completed.")