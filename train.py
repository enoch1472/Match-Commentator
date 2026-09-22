import os
import gc
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if "__file__" in locals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(BASE_DIR, "football_model_lora")
DATASET_PATH = os.path.join(BASE_DIR, "train.jsonl")

if not os.path.exists(DATASET_PATH):
    print("Dataset not found. Generating...")
    from fetch_and_prepare_dataset import build_dataset
    build_dataset()

has_cuda = torch.cuda.is_available()
print("=" * 60)
print(f"Hardware: {'GPU (CUDA)' if has_cuda else 'CPU'}")
print(f"Model:    {MODEL_ID}")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_dtype = torch.float32 if not has_cuda else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
        device_map="auto" if has_cuda else None
    )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

dataset = load_dataset("json", data_files={"train": DATASET_PATH})

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=5e-4,
    logging_steps=5,
    save_strategy="no",
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=False,
    report_to="none",
    dataloader_num_workers=0,
    use_cpu=not has_cuda
)

def train_model():
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "peft_config": peft_config,
        "args": training_args,
    }
    
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    print(">>> Starting LoRA Fine-Tuning (5 Epochs)...")
    trainer.train()

    print(f">>> Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(">>> Training complete! Ready for compare.py")

if __name__ == "__main__":
    train_model()
