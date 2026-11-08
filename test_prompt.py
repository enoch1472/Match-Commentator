# -*- coding: utf-8 -*-
"""
Premier League Match Report Generator - Interactive Terminal Tester
Supports entering custom stats, inspecting live fixtures from football-data.org,
and toggling between summary formats (one-sentence, bullets, tweet).
"""

import os
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(BASE_DIR, "football_model_lora")
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "8cb8eb39b3914b2fbd3f515f8c62f36e")

print("Loading model and LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
has_cuda = torch.cuda.is_available()
model_dtype = torch.float16 if has_cuda else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    device_map="auto" if has_cuda else None
)

if os.path.exists(ADAPTER_PATH):
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Fine-tuned LoRA model loaded successfully!")
else:
    model = base_model
    print("[Warning] LoRA adapter not found. Using baseline model.")

def summarize(stat_prompt: str, format_mode: str = "sentence") -> str:
    if format_mode == "bullets":
        system_content = "You are a sports analyst. Write 3 concise bullet points summarizing the match result, scoreline, and key outcome without hallucinating."
        max_tokens = 90
    elif format_mode == "tweet":
        system_content = "You are a Premier League social media editor. Write an engaging, factual tweet with the score, winner, and hashtags strictly based on the provided stats."
        max_tokens = 60
    else:
        system_content = "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline and teams."
        max_tokens = 45

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": stat_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = "cuda" if has_cuda else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def fetch_live_sample() -> str:
    """Fetch the latest finished fixture from football-data.org as a sample."""
    try:
        url = "https://api.football-data.org/v4/competitions/PL/matches?season=2024&status=FINISHED"
        r = requests.get(url, headers={"X-Auth-Token": FOOTBALL_DATA_API_KEY}, timeout=8)
        if r.status_code == 200:
            matches = r.json().get("matches", [])
            if matches:
                m = matches[-1]
                h = m.get("homeTeam", {}).get("shortName", "Home")
                a = m.get("awayTeam", {}).get("shortName", "Away")
                score = m.get("score", {}).get("fullTime", {})
                return f"{h} {score.get('home', 0)}-{score.get('away', 0)} {a}."
    except Exception:
        pass
    return "Arsenal 2-1 Chelsea. Scorers: Arsenal: Saka 24', Havertz 88'; Chelsea: Palmer 61'."

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" Premier League Match Report Generator - Interactive Tester")
    print(" Formats: [1] One-Sentence Recap | [2] Bullets | [3] Social Media Tweet")
    print(" Commands: 'live' (pull from API), 'mode <1|2|3>', 'q' or 'exit' to quit")
    print("=" * 65)

    mode = "sentence"
    sample = "Arsenal 2-1 Chelsea. Scorers: Arsenal: Saka 24', Havertz 88'; Chelsea: Palmer 61'."
    print(f"\nExample Input:  {sample}")
    print(f"Model Summary:  {summarize(sample, mode)}\n")

    while True:
        try:
            user_input = input(f"Enter match stats [{mode}] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                print("Exiting.")
                break
            if user_input.lower() == "live":
                user_input = fetch_live_sample()
                print(f"--> Live Fixture: {user_input}")
            elif user_input.lower().startswith("mode"):
                parts = user_input.split()
                if len(parts) > 1:
                    if parts[1] == "2":
                        mode = "bullets"
                    elif parts[1] == "3":
                        mode = "tweet"
                    else:
                        mode = "sentence"
                    print(f"--> Switched format to: {mode}")
                    continue

            output = summarize(user_input, mode)
            print(f"--> Summary: {output}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
