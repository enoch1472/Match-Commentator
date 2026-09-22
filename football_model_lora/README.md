Football Summary LoRA Model
Enoch Obikoya

This folder contains the LoRA adapter weights I trained for generating short Premier League match reports.

Overview

I built this project to learn and use LoRA in practice, fine tuning these weights on Qwen2.5 0.5B Instruct to stop the base model from guessing wrong scores or making up fake match events. The adapter focuses specifically on turning raw scorelines and goalscorers into accurate single sentence summaries.

Usage

You can load the adapter in Python like this:

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "football_model_lora")

Training Setup

I trained the adapter using PEFT and SFTTrainer on 212 match examples over 5 epochs. Only the attention projection layers were updated, keeping the base model lightweight and quick to run.