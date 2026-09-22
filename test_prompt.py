import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(BASE_DIR, "football_model_lora")

print("Loading model and LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
has_cuda = torch.cuda.is_available()
model_dtype = torch.float16 if has_cuda else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=model_dtype,
    device_map="auto" if has_cuda else None
)

if os.path.exists(ADAPTER_PATH):
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Fine-tuned LoRA model loaded successfully!")
else:
    model = base_model
    print("[Warning] LoRA adapter not found. Using baseline model.")

def summarize(stat_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline and teams."
        },
        {"role": "user", "content": stat_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = "cuda" if has_cuda else "cpu"
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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Premier League Match Report Generator - Interactive Tester")
    print(" Type or paste any match stats below. (Type 'q' or 'exit' to quit)")
    print("=" * 60)
    
    sample = "Arsenal 2-1 Chelsea. Scorers: Arsenal: Saka 24', Havertz 88'; Chelsea: Palmer 61'."
    print(f"\nExample Input:  {sample}")
    print(f"Model Summary:  {summarize(sample)}\n")
    
    while True:
        try:
            user_input = input("Enter match stats > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                print("Exiting.")
                break
            
            output = summarize(user_input)
            print(f"--> Summary: {output}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
