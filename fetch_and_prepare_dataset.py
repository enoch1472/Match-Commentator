# -*- coding: utf-8 -*-
"""
Football Match Report Generator - Step 1: Dataset Fetcher & Preparer
Generates strictly verified ground-truth summaries with exact scorelines and winners.
"""

import os
import json
import random
import re
import requests

if "__file__" in locals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

TRAIN_PATH = os.path.join(BASE_DIR, "train.jsonl")
TEST_PATH = os.path.join(BASE_DIR, "test.jsonl")

TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Liverpool", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle", "Nottingham Forest", "Sheffield United", "Tottenham", "West Ham", "Wolves"
]

PLAYERS = {
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
    "Luton Town": ["Morris", "Adebayo", "Chong", "Barkley", "Ogbene"],
    "Sheffield United": ["Archer", "Brereton Diaz", "McBurnie", "Hamer", "Bogle"]
}

def generate_verified_fixture():
    home_team, away_team = random.sample(TEAMS, 2)
    home_score = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 14, 6])[0]
    away_score = random.choices([0, 1, 2, 3], weights=[25, 40, 25, 10])[0]

    def get_scorers(team, count):
        if count == 0:
            return ""
        pool = PLAYERS.get(team, ["Striker", "Winger", "Midfielder"])
        selected = random.choices(pool, k=count)
        minutes = sorted(random.sample(range(1, 91), count))
        parts = [f"{player} {minute}'" for player, minute in zip(selected, minutes)]
        return ", ".join(parts)

    h_scorers = get_scorers(home_team, home_score)
    a_scorers = get_scorers(away_team, away_score)

    # Build exact input string;
    scorer_segments = []
    if h_scorers:
        scorer_segments.append(f"{home_team}: {h_scorers}")
    if a_scorers:
        scorer_segments.append(f"{away_team}: {a_scorers}")

    if scorer_segments:
        stat_input = f"{home_team} {home_score}-{away_score} {away_team}. Scorers: {'; '.join(scorer_segments)}."
    else:
        stat_input = f"{home_team} {home_score}-{away_score} {away_team}."

    # Build strict, non-hallucinated ground-truth summary
    if home_score > away_score:
        if away_score == 0:
            summary = f"{home_team} earned a comfortable {home_score}-0 clean-sheet victory over {away_team}."
        elif home_score - away_score == 1:
            summary = f"{home_team} edged past {away_team} with a hard-fought {home_score}-{away_score} win."
        else:
            summary = f"{home_team} defeated {away_team} {home_score}-{away_score} with a dominant performance."
    elif away_score > home_score:
        if home_score == 0:
            summary = f"{away_team} secured an impressive {away_score}-0 away win against {home_team}."
        elif away_score - home_score == 1:
            summary = f"{away_team} claimed a tight {away_score}-{home_score} away victory over {home_team}."
        else:
            summary = f"{away_team} beat {home_team} {away_score}-{home_score} on the road."
    else:
        if home_score == 0:
            summary = f"{home_team} and {away_team} played out a scoreless 0-0 draw."
        else:
            summary = f"{home_team} and {away_team} shared the points in an entertaining {home_score}-{away_score} draw."

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline and teams."
            },
            {
                "role": "user",
                "content": stat_input
            },
            {
                "role": "assistant",
                "content": summary
            }
        ]
    }

def build_dataset():
    random.seed(42)
    dataset_pairs = [generate_verified_fixture() for _ in range(250)]

    split_idx = int(len(dataset_pairs) * 0.85)
    train_data = dataset_pairs[:split_idx]
    test_data = dataset_pairs[split_idx:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[Done] Generated 250 verified match pairs.")
    print(f"-> Saved {len(train_data)} training samples to: {TRAIN_PATH}")
    print(f"-> Saved {len(test_data)} test samples to: {TEST_PATH}")

if __name__ == "__main__":
    build_dataset()
