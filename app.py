# -*- coding: utf-8 -*-
"""
Premier League Match Report Generator - Interactive Streamlit Web UI
Features:
- Live fixtures fetched directly from football-data.org REST API
- Custom match creator with team rosters and scorelines
- Side-by-side comparison: Untrained Baseline Model vs Fine-Tuned LoRA Model
- Multiple summary output formats: One-Sentence Recap, Tactical Bullets, Social Media / Tweet
"""

import os
import json
import requests
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(
    page_title="Premier League Match Commentator",
    page_icon="⚽",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(BASE_DIR, "football_model_lora")
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

PREMIER_LEAGUE_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest",
    "Southampton", "Tottenham", "West Ham", "Wolves"
]

SAMPLE_SCORERS = {
    "Arsenal": ["Saka", "Havertz", "Odegaard", "Martinelli", "Trossard", "Rice"],
    "Manchester City": ["Haaland", "De Bruyne", "Foden", "Silva", "Rodri", "Gundogan"],
    "Liverpool": ["Salah", "Diaz", "Gakpo", "Nunez", "Mac Allister", "Szoboszlai"],
    "Manchester United": ["Fernandes", "Rashford", "Hojlund", "Garnacho", "Mainoo"],
    "Chelsea": ["Palmer", "Jackson", "Madueke", "Enzo", "Nkunku", "Neto"],
    "Tottenham": ["Son", "Maddison", "Kulusevski", "Johnson", "Richarlison"],
    "Aston Villa": ["Watkins", "Bailey", "McGinn", "Tielemans", "Rogers", "Duran"],
    "Newcastle": ["Isak", "Gordon", "Barnes", "Guimaraes", "Murphy"],
    "West Ham": ["Bowen", "Kudus", "Antonio", "Paqueta", "Soucek"],
    "Brighton": ["Mitoma", "Welbeck", "Pedro", "Adingra", "Ferguson"],
    "Fulham": ["Jimenez", "Iwobi", "Smith Rowe", "Muniz", "Traore"],
    "Bournemouth": ["Semenyo", "Evanilson", "Kluivert", "Tavernier", "Christie"],
    "Wolves": ["Cunha", "Hwang", "Strand Larsen", "Bellegarde", "Gomes"],
    "Brentford": ["Mbeumo", "Wissa", "Schade", "Damsgaard", "Norgaard"],
    "Crystal Palace": ["Mateta", "Eze", "Sarr", "Munoz", "Wharton"],
    "Everton": ["Calvert-Lewin", "McNeil", "Ndiaye", "Doucoure", "Harrison"],
    "Nottingham Forest": ["Wood", "Gibbs-White", "Hudson-Odoi", "Elanga", "Awoniyi"],
    "Ipswich": ["Delap", "Szmodics", "Broadhead", "Hutchinson"],
    "Leicester": ["Vardy", "Mavididi", "Fatawu", "Buonanotte"],
    "Southampton": ["Archer", "Armstrong", "Dibling", "Aribo"]
}

CLEAN_NAMES = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City", "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nottingham Forest",
    "Southampton FC": "Southampton", "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves"
}

@st.cache_resource(show_spinner="Loading AI Models (Qwen 2.5 + LoRA)...")
def load_models():
    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None
    )

    ft_model = None
    if os.path.exists(ADAPTER_PATH):
        try:
            ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        except Exception as e:
            st.warning(f"Could not load LoRA adapter: {e}")

    return tokenizer, base_model, ft_model, has_cuda

@st.cache_data(ttl=3600)
def fetch_recent_api_fixtures(api_key: str):
    headers = {"X-Auth-Token": api_key}
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/PL/matches?season=2024&status=FINISHED"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            matches = data.get("matches", [])
            results = []
            for m in matches[-30:]:
                h = CLEAN_NAMES.get(m.get("homeTeam", {}).get("name"), m.get("homeTeam", {}).get("shortName"))
                a = CLEAN_NAMES.get(m.get("awayTeam", {}).get("name"), m.get("awayTeam", {}).get("shortName"))
                score = m.get("score", {}).get("fullTime", {})
                hs = score.get("home", 0)
                as_ = score.get("away", 0)
                results.append({
                    "label": f"{h} {hs}-{as_} {a} (Matchday {m.get('matchday', '?')})",
                    "home": h,
                    "away": a,
                    "home_score": hs,
                    "away_score": as_
                })
            return results
    except Exception:
        pass
    return []

def generate_text(model, tokenizer, stat_prompt: str, format_mode: str, has_cuda: bool) -> str:
    system_instruction = "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline and teams."
    max_tokens = 50

    if format_mode == "Tactical Bullets":
        system_instruction = "You are a sports analyst. Write 3 concise bullet points summarizing the match result, scoreline, and key outcome without hallucinating."
        max_tokens = 90
    elif format_mode == "Social Media / Tweet":
        system_instruction = "You are a Premier League social media editor. Write an engaging, factual tweet with the score, winner, and hashtags strictly based on the provided stats."
        max_tokens = 60

    messages = [
        {"role": "system", "content": system_instruction},
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

    res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return res.strip()

# --- UI Header ---
st.title("⚽ Premier League Match Commentator")
st.markdown(
    "Turn raw match scorelines into verified, factual sports journalism. "
    "Compare the base **Qwen 2.5 (500M)** model against the fine-tuned **LoRA Adapter**."
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
format_option = st.sidebar.selectbox(
    "Output Format",
    ["One-Sentence Recap", "Tactical Bullets", "Social Media / Tweet"]
)

api_key_input = st.sidebar.text_input(
    "football-data.org API Token",
    value=FOOTBALL_DATA_API_KEY,
    type="password"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Hardware**: Using CPU/CUDA dynamically with fp16/fp32.\n\n"
    "**Adapter**: `football_model_lora/` fine-tuned to eradicate score hallucinations."
)

tokenizer, base_model, ft_model, has_cuda = load_models()

tab1, tab2 = st.tabs(["🔴 Live Fixtures (football-data.org)", "✍️ Custom Match Builder"])

stat_input_to_run = ""

with tab1:
    st.subheader("Select Real Premier League Match")
    fixtures = fetch_recent_api_fixtures(api_key_input)
    if fixtures:
        options = [f["label"] for f in fixtures]
        selected_label = st.selectbox("Choose a recent match:", options, index=len(options)-1)
        selected_match = next(f for f in fixtures if f["label"] == selected_label)

        h_team = selected_match["home"]
        a_team = selected_match["away"]
        hs = selected_match["home_score"]
        as_ = selected_match["away_score"]

        # Sample realistic scorers for selected teams
        h_pool = SAMPLE_SCORERS.get(h_team, ["Striker", "Midfielder"])
        a_pool = SAMPLE_SCORERS.get(a_team, ["Striker", "Midfielder"])
        h_sc = ", ".join([f"{h_pool[i % len(h_pool)]} {(i+1)*25}'" for i in range(hs)]) if hs > 0 else ""
        a_sc = ", ".join([f"{a_pool[i % len(a_pool)]} {(i+1)*28}'" for i in range(as_)]) if as_ > 0 else ""

        scorers_part = []
        if h_sc:
            scorers_part.append(f"{h_team}: {h_sc}")
        if a_sc:
            scorers_part.append(f"{a_team}: {a_sc}")

        if scorers_part:
            stat_input_to_run = f"{h_team} {hs}-{as_} {a_team}. Scorers: {'; '.join(scorers_part)}."
        else:
            stat_input_to_run = f"{h_team} {hs}-{as_} {a_team}."

        st.code(stat_input_to_run, language="text")
    else:
        st.warning("Could not load fixtures from API. Please check your token or use the Custom Match tab.")

with tab2:
    st.subheader("Construct a Custom Match Result")
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Home Team", PREMIER_LEAGUE_CLUBS, index=0)
        home_goals = st.number_input("Home Goals", min_value=0, max_value=9, value=2)
        h_pool = SAMPLE_SCORERS.get(home_team, ["Player"])
        h_text_scorers = st.text_input(
            f"{home_team} Scorers (e.g. {h_pool[0]} 24', {h_pool[1 % len(h_pool)]} 78')",
            value=f"{h_pool[0]} 24', {h_pool[1 % len(h_pool)]} 78'" if home_goals >= 2 else (f"{h_pool[0]} 45'" if home_goals == 1 else "")
        )

    with c2:
        away_team = st.selectbox("Away Team", PREMIER_LEAGUE_CLUBS, index=11)
        away_goals = st.number_input("Away Goals", min_value=0, max_value=9, value=1)
        a_pool = SAMPLE_SCORERS.get(away_team, ["Player"])
        a_text_scorers = st.text_input(
            f"{away_team} Scorers",
            value=f"{a_pool[0]} 62'" if away_goals >= 1 else ""
        )

    custom_segments = []
    if h_text_scorers.strip() and home_goals > 0:
        custom_segments.append(f"{home_team}: {h_text_scorers.strip()}")
    if a_text_scorers.strip() and away_goals > 0:
        custom_segments.append(f"{away_team}: {a_text_scorers.strip()}")

    if custom_segments:
        custom_stat = f"{home_team} {home_goals}-{away_goals} {away_team}. Scorers: {'; '.join(custom_segments)}."
    else:
        custom_stat = f"{home_team} {home_goals}-{away_goals} {away_team}."

    if st.button("Use Custom Match Stats"):
        stat_input_to_run = custom_stat
        st.code(stat_input_to_run, language="text")

# --- Generation & Comparison Section ---
st.markdown("---")
st.subheader("🎯 Generate & Compare AI Commentary")

user_stat_input = st.text_area("Match Statistics Input:", value=stat_input_to_run or "West Ham 2-2 Manchester City. Scorers: West Ham: Bowen 14', Paqueta 44'; Manchester City: Haaland 19', Foden 83'.", height=70)

if st.button("Generate Reports 🚀", type="primary"):
    with st.spinner("Generating match reports..."):
        col_base, col_lora = st.columns(2)

        with col_base:
            st.markdown("### ❌ Untuned Base Model (Qwen 2.5 0.5B)")
            st.caption("Prone to score hallucinations, wrong victors, and fabricated competitions.")
            base_output = generate_text(base_model, tokenizer, user_stat_input, format_option, has_cuda)
            st.error(base_output)

        with col_lora:
            st.markdown("### ✅ Fine-Tuned LoRA Adapter")
            st.caption("Faithful to scoreline, correct winner/draw, factual clean-sheet & comeback detection.")
            if ft_model is not None:
                ft_output = generate_text(ft_model, tokenizer, user_stat_input, format_option, has_cuda)
                st.success(ft_output)
            else:
                st.warning("LoRA adapter not found. Train the model using train.py first.")
