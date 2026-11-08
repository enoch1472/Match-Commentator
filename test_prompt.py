# -*- coding: utf-8 -*-
import os
import re
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(BASE_DIR, "football_model_lora")
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")

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

def ensure_scorers_in_report(summary: str, stat_prompt: str) -> str:
    if "Scorers:" not in stat_prompt:
        return summary

    scorers_part = stat_prompt.split("Scorers:", 1)[1].strip().rstrip(".")
    team_segments = [s.strip() for s in scorers_part.split(";") if s.strip()]

    parsed = {}
    for seg in team_segments:
        if ":" in seg:
            team, p_str = seg.split(":", 1)
            raw = [re.sub(r"\s*\d+[\'’]", "", p).strip() for p in p_str.split(",") if p.strip()]
            raw = [n for n in raw if n]
            if raw:
                parsed[team.strip()] = raw

    if not parsed:
        return summary

    all_players = [p for names in parsed.values() for p in names]
    if any(p.lower() in summary.lower() for p in all_players):
        return summary

    def join_names(names):
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    base = summary.rstrip(".")
    if len(parsed) == 2:
        teams = list(parsed.keys())
        t1, t2 = teams[0], teams[1]
        p1 = join_names(parsed[t1])
        p2 = join_names(parsed[t2])
        return f"{base}, with {p1} on target for {t1} and {p2} scoring for {t2}."
    elif len(parsed) == 1:
        team = list(parsed.keys())[0]
        p = join_names(parsed[team])
        return f"{base}, courtesy of goals from {p}."

    return summary

def summarize(stat_prompt: str) -> str:
    system_content = "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline, teams, and goalscorers."
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
            max_new_tokens=65,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return ensure_scorers_in_report(response, stat_prompt)

def fetch_live_sample() -> str:
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
    print(" Type match stats, or 'live' to pull latest from API.")
    print(" Type 'q' or 'exit' to quit.")
    print("=" * 65)

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
            if user_input.lower() == "live":
                user_input = fetch_live_sample()
                print(f"--> Live Fixture: {user_input}")

            output = summarize(user_input)
            print(f"--> Summary: {output}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
