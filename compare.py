import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if "__file__" in locals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(BASE_DIR, "football_model_lora")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test.jsonl")

test_cases = [
    ("Arsenal 2-1 Nottingham Forest. Scorers: Arsenal: Saka 24', Havertz 88'; Nottingham Forest: Gibbs-White 45'.", "Arsenal 2-1 Nottingham Forest"),
    ("West Ham 3-2 Nottingham Forest. Scorers: West Ham: Bowen 12', Kudus 45', Antonio 78'; Nottingham Forest: Wood 33', Elanga 80'.", "West Ham 3-2 Nottingham Forest"),
    ("Bournemouth 2-3 Fulham. Scorers: Bournemouth: Semenyo 14', Evanilson 67'; Fulham: Jimenez 22', 54', Iwobi 81'.", "Bournemouth 2-3 Fulham"),
    ("West Ham 1-0 Brighton. Scorers: West Ham: Bowen 47'.", "West Ham 1-0 Brighton")
]

if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
        if len(lines) >= 4:
            test_cases = [(item["messages"][1]["content"], item["messages"][1]["content"].split(".")[0]) for item in lines[:4]]

print(f"Loading Base Model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

has_cuda = torch.cuda.is_available()
model_dtype = torch.float32 if not has_cuda else torch.float16

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None
    )
except TypeError:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
        device_map="auto" if has_cuda else None
    )

def generate_summary(model, stat_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline and teams."
        },
        {"role": "user", "content": stat_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = base_model.device if hasattr(base_model, "device") else ("cuda" if has_cuda else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_comparison():
    print("\n" + "=" * 80)
    print("1. EVALUATING BASELINE MODEL")
    print("=" * 80)
    base_results = []
    for stat_prompt, label in test_cases:
        res = generate_summary(base_model, stat_prompt)
        base_results.append(res)
        print(f"\nInput:    {stat_prompt}")
        print(f"Baseline: {res}")

    if not os.path.exists(ADAPTER_PATH):
        print(f"\n[Notice] Adapter folder '{ADAPTER_PATH}' not found. Run 'train.py' first.")
        return

    print("\n" + "=" * 80)
    print("2. EVALUATING ADAPTED MODEL (AFTER LoRA)")
    print("=" * 80)
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE VERIFICATION TABLE:")
    print("=" * 80)
    for i, (stat_prompt, label) in enumerate(test_cases):
        ft_res = generate_summary(ft_model, stat_prompt)
        print(f"\nFixture {i+1}: [{label}]")
        print(f"Input Stats:     {stat_prompt}")
        print(f"Baseline Output: {base_results[i]}")
        print(f"Adapted Output:  {ft_res}")

if __name__ == "__main__":
    run_comparison()
