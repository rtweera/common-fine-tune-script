import unsloth
import subprocess
import sys
from dotenv import load_dotenv
from typing import Literal, Union
import os
import torch
from datasets import load_dataset
from datetime import datetime
import pytz
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import wandb
from huggingface_hub import login
from unsloth import is_bfloat16_supported
import yaml

from logger import Logger
from logger_callback import LoggerCallback

class FineTuneQwenCommon:
    qwen_models = {
        "0.5b": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "3b": "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
        "7b": "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit", # unsloth-bnb-4bit are selectively quantized for more accuracy
    } # More models at https://huggingface.co/unsloth

    def __init__(self, config_path="config.yaml"):
        self.logger = Logger()
        self.logger.info("Initializing FineTuneQwenCommon...")

        self.logger.info("Loading configuration from {}".format(config_path))
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if load_dotenv(override=True, verbose=True):
            self.logger.info("Environment variables loaded successfully.")
        else:
            self.logger.error("Failed to load environment variables. Please check your .env file.")
            raise EnvironmentError("Failed to load environment variables.")

        # Configurations
        self.model_key = config.get("model_key", "0.5b")
        self.model_name = self.qwen_models[self.model_key]
        self.max_seq_length = config.get("max_seq_length", 2048)
        self.dtype = config.get("dtype", None)
        self.load_in_4bit = config.get("load_in_4bit", True)
        self.rank = config.get("rank", 16)
        self.dataset_id = config.get("dataset_id")
        self.dataset_columns = config.get("dataset_columns", ["question", "answer", "topic"])
        self.user_column = config.get("user_column")
        self.assistant_column = config.get("assistant_column")
        self.system_column = config.get("system_column")
        self.system_prompt = config.get("system_prompt")
        self.run_name_prefix = config.get("run_name_prefix")
        self.run_name_suffix = config.get("run_name_suffix", "LoRA")
        self.project_name = config.get("project_name", "choreo-doc-assistant-lora")
        self.device_batch_size = config.get("device_batch_size", 4)
        self.grad_accumulation = config.get("grad_accumulation", 4)
        self.epochs = config.get("epochs", 30)
        self.logger_log_interval_seconds = config.get("logger_log_interval_seconds", 60)
        self.learning_rate = config.get("learning_rate", 2e-4)
        self.warmup_steps = config.get("warmup_steps", 5)
        self.optim = config.get("optim", "paged_adamw_8bit")
        self.weight_decay = config.get("weight_decay", 0.01)
        self.lr_scheduler_type = config.get("lr_scheduler_type", "linear")
        self.seed = config.get("seed", 3407)
        self.logging_steps = config.get("logging_steps", 1)
        self.logging_first_step = config.get("logging_first_step", True)
        self.save_steps = config.get("save_steps", 20)
        self.save_total_limit = config.get("save_total_limit", 4)
        self.push_to_hub = config.get("push_to_hub", True)
        self.packing = config.get("packing", False)
        self.dataset_num_proc = config.get("dataset_num_proc", 4)
        self.target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                            "gate_proj", "up_proj", "down_proj"])
        self.lora_alpha = config.get("lora_alpha", 16)
        self.lora_dropout = config.get("lora_dropout", 0)
        self.bias = config.get("bias", "none")
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", "unsloth")
        self.use_rslora = config.get("use_rslora", False)
        self.loftq_config = config.get("loftq_config", None)

        self.instruction_part = config.get("instruction_part", "<|im_start|>user\n")
        self.response_part = config.get("response_part", "<|im_start|>assistant\n")

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_base_model_and_tokenizer()
        self.model = self._get_peft_model()

    def _load_base_model_and_tokenizer(self):
        self.logger.info(f"Loading base model and tokenizer for {self.model_name}...")
        return FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            token = os.getenv("HF_TOKEN")
        )

    def _get_peft_model(self):
        self.logger.info(f"Converting model to PEFT with rank {self.rank}...")
        return FastLanguageModel.get_peft_model(
            self.model,
            r = self.rank,
            target_modules = self.target_modules,
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout,
            bias = self.bias,
            use_gradient_checkpointing = self.use_gradient_checkpointing,
            random_state = self.seed,
            use_rslora = self.use_rslora,
            loftq_config = self.loftq_config,
        )

    def _convert_to_conversations(self, example):
        if self.system_prompt is not None:
            instruction_part = self.system_prompt.strip()
        elif self.system_column is not None:
            instruction_part = example[self.system_column].strip()
        else:
            instruction_part = None
        user_part = example['question'].strip()
        assistant_part = example['answer'].strip()
        output = {
            "conversations": [
                {"role": "user", "content": user_part},
                {"role": "assistant", "content": assistant_part}
            ]
        }
        if instruction_part is not None:
            output["conversations"].insert(0, {"role": "system", "content": instruction_part})
        return output

    def _formatting_prompts_func(self, examples):
        self.logger.info("Formatting prompts for training...")
        convos = examples['conversations']
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
        

    def _handle_wandb_setup(self):
        self.logger.info("Setting up Weights & Biases...")
        wandb.login(key=os.getenv('WANDB_TOKEN'))

        # Run name logic
        MODEL_SAVENAME = self.model_name.split('/')[-1]
        now_utc = datetime.now(pytz.utc)
        now_colombo = now_utc.astimezone(pytz.timezone('Asia/Colombo'))
        time_str = now_colombo.strftime('%Y-%b-%d_%H-%M-%S')
        run_name = f'{time_str}_{MODEL_SAVENAME}'
        if self.run_name_prefix is not None:
            run_name = f'{self.run_name_prefix}_{run_name}'
        if self.run_name_suffix is not None:
            run_name = f'{run_name}_{self.run_name_suffix}'
        self.logger.info(f"Run name set: {run_name}")

        wandb.init(project=self.project_name, name=run_name)
        self.logger.info("Weights & Biases setup complete.")
        return run_name

    def _login_huggingface(self):
        self.logger.info("Logging in to Hugging Face...")
        login(token=os.getenv("HF_TOKEN"))
        self.logger.info("Hugging Face login successful.")

    def run(self):
        # Load and process dataset
        self._login_huggingface()
        self.logger.info("Loading and processing dataset...")
        dataset = load_dataset(self.dataset_id, split = "train")
        dataset = dataset.map(self._convert_to_conversations, remove_columns=self.dataset_columns, batched=False)
        dataset = dataset.map(self._formatting_prompts_func, batched=True)

        self.run_name = self._handle_wandb_setup()

        self.logger.info("Configuring trainer...")
        # Training
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
            dataset_num_proc = self.dataset_num_proc,
            packing = self.packing,
            args = TrainingArguments(
                per_device_train_batch_size = self.device_batch_size,
                gradient_accumulation_steps = self.grad_accumulation,
                warmup_steps = self.warmup_steps,
                num_train_epochs = self.epochs,
                learning_rate = self.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.logging_steps,
                logging_first_step=self.logging_first_step,
                optim = self.optim,
                weight_decay = self.weight_decay,
                lr_scheduler_type = self.lr_scheduler_type,
                seed = self.seed,
                output_dir = self.run_name,
                report_to = "wandb",
                save_steps=self.save_steps,
                save_total_limit=self.save_total_limit,
                push_to_hub=self.push_to_hub,
                hub_model_id=self.run_name
            ),
            callbacks = [LoggerCallback(logger=self.logger, log_interval_seconds=60)]
        )

        self.logger.info("Setting the loss for generated tokens only...")
        trainer = train_on_responses_only(
            trainer,
            instruction_part = self.instruction_part,
            response_part = self.response_part,
        )

        self.logger.info("Starting training...")
        trainer_stats = trainer.train()
        self.logger.info("Training completed successfully.")
        return trainer_stats

if __name__ == "__main__":
    try:
        fine_tune = FineTuneQwenCommon()
        fine_tune.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)  # Exit with a non-zero status code to indicate failure

