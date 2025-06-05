import unsloth
from unsloth import FastLanguageModel
import torch
import yaml
from dotenv import load_dotenv
import os
from transformers import TextStreamer
from datasets import load_dataset

from logger import Logger 


class RunnerQwenCommon:
    def __init__(self, config_path='running-config.yaml'):
        # logger
        self.logger = Logger()
        self.logger.info("Initializing RunnerQwenCommon...")
        
        # load configuration and environment variables
        self.logger.info("Loading configuration from {}".format(config_path))
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if load_dotenv(override=True, verbose=True):
            self.logger.info("Environment variables loaded successfully.")
        else:
            self.logger.error("Failed to load environment variables. Please check your .env file.")
            raise EnvironmentError("Failed to load environment variables.")

        # Configurations assigned from the config file
        self.logger.info("Loading configuration from: {}".format(config))
        self.model_name = config.get('model_id', None)
        self.max_seq_length = config.get('max_seq_length', 2048)
        self.dtype = config.get('dtype', None)
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.dataset_id = config.get('dataset_id', None)
        self.dataset_columns = config.get('dataset_columns', ["question", "answer", "topic"])
        self.user_column = config.get('user_column', "question")
        self.assistant_column = config.get('assistant_column', "answer")
        self.system_column = config.get('system_column', "topic")
        self.system_prompt = config.get('system_prompt', None)
        self.use_system_prompt = config.get('use_system_prompt', False)
        self.max_new_tokens = config.get('max_new_tokens', 2048)
        self.use_cache = config.get('use_cache', True)
        self.temperature = config.get('temperature', 1.5)
        self.min_p = config.get('min_p', 0.1)
        self.output_file = config.get('output_file', 'output.csv')

        # load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            token = os.getenv("HF_TOKEN")
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        self.logger.info("Model and tokenizer loaded successfully.")

        # set up text streamer
        self.logger.info("Setting up text streamer for model output...")
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.logger.info("Initialization complete.")

    def _get_messages(self, user_prompt):
        """
        Prepare messages for the model.
        """
        messages = []
        if self.use_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        tokenized_msg = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.model.device)
        return tokenized_msg

    def stream_sample(self, sample_question):
        """
        Stream the model's response to a sample question.
        """
        self.logger.info("Streaming response for sample question: {}".format(sample_question))
        tokenized_msg = self._get_messages(sample_question)
        
        self.model.generate(
            input_ids=tokenized_msg,
            attention_mask=tokenized_msg.attention_mask,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
            use_cache=self.use_cache,
            temperature=self.temperature,
            min_p=self.min_p,
        )
        self.logger.info("Response streaming completed.")
    
    def generate_response(self):
        user_prompts = [
    "How to deploy a webapp?",
    "How to resolve \"\"Module not found error\"\" during the deployment of a Python project?",
]
        data = load_dataset(self.dataset_id, split='train')
        questions = data[self.user_column]

        with open(self.output_file, "w", encoding="utf-8") as f:
            for prompt in questions:
                inputs = self._get_messages(prompt)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=self.use_cache,
                    temperature=self.temperature,
                    min_p=self.min_p
                )
                output_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
