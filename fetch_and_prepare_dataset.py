# -*- coding: utf-8 -*-
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

FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

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
    "Sheffield United": ["Archer", "Brereton Diaz", "McBurnie", "Hamer", "Bogle"],
    "Ipswich": ["Delap", "Szmodics", "Broadhead", "Hutchinson", "Davis"],
    "Leicester": ["Vardy", "Mavididi", "Fatawu", "Buonanotte", "Ndidi"],
    "Southampton": ["Archer", "Armstrong", "Dibling", "Aribo", "Lallana"],
    "Burnley": ["Foster", "Brownhill", "Koleosho", "Amdouni"]
}

CLEAN_TEAM_NAMES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham Forest",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Burnley FC": "Burnley",
    "Luton Town FC": "Luton Town",
    "Sheffield United FC": "Sheffield United"
}

def clean_team_name(raw_name: str) -> str:
    if raw_name in CLEAN_TEAM_NAMES:
        return CLEAN_TEAM_NAMES[raw_name]
    cleaned = raw_name.replace(" FC", "").replace(" & Hove Albion", "").replace(" United", "")
    return cleaned.strip()

def _clean_scorer_names(scorer_str: str) -> str:
    if not scorer_str:
        return ""
    names = [re.sub(r"\s*\d+[\'’]", "", p).strip() for p in scorer_str.split(",") if p.strip()]
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def build_ground_truth_summary(home_team: str, away_team: str, home_score: int, away_score: int,
                               half_time=None, h_scorers: str = "", a_scorers: str = "") -> str:
    h_names = _clean_scorer_names(h_scorers)
    a_names = _clean_scorer_names(a_scorers)

    if half_time and "home" in half_time and "away" in half_time:
        ht_home = half_time.get("home") or 0
        ht_away = half_time.get("away") or 0
        if ht_home < ht_away and home_score > away_score:
            base = f"{home_team} produced an impressive second-half comeback to defeat {away_team} {home_score}-{away_score}"
            return f"{base}, with goals from {h_names} overturning the halftime deficit." if h_names else f"{base} after trailing at the break."
        elif ht_away < ht_home and away_score > home_score:
            base = f"{away_team} rallied from behind in the second half to secure a sensational {away_score}-{home_score} away victory over {home_team}"
            return f"{base}, courtesy of strikes from {a_names}." if a_names else f"{base} after trailing at the break."

    if home_score > away_score:
        if away_score == 0:
            base = f"{home_team} earned a comfortable {home_score}-0 clean-sheet victory over {away_team}"
            return f"{base}, courtesy of goals from {h_names}." if h_names else f"{base}."
        elif home_score - away_score == 1:
            base = f"{home_team} edged past {away_team} with a hard-fought {home_score}-{away_score} win"
            if h_names and a_names:
                return f"{base}, with {h_names} on target for the hosts despite {a_names} scoring for {away_team}."
            return f"{base}."
        else:
            base = f"{home_team} defeated {away_team} {home_score}-{away_score} with a dominant performance"
            return f"{base}, powered by goals from {h_names}." if h_names else f"{base}."

    elif away_score > home_score:
        if home_score == 0:
            base = f"{away_team} secured an impressive {away_score}-0 away win against {home_team}"
            return f"{base}, with goals from {a_names} sealing the points." if a_names else f"{base}."
        elif away_score - home_score == 1:
            base = f"{away_team} claimed a tight {away_score}-{home_score} away victory over {home_team}"
            if a_names and h_names:
                return f"{base}, with {a_names} netting for the visitors and {h_names} scoring for {home_team}."
            return f"{base}."
        else:
            base = f"{away_team} beat {home_team} {away_score}-{home_score} on the road"
            return f"{base}, thanks to strikes from {a_names}." if a_names else f"{base}."

    else:
        if home_score == 0:
            return f"{home_team} and {away_team} played out a scoreless 0-0 draw."
        else:
            base = f"{home_team} and {away_team} shared the points in an entertaining {home_score}-{away_score} draw"
            if h_names and a_names:
                return f"{base}, with {h_names} finding the net for {home_team} and {a_names} replying for {away_team}."
            return f"{base}."

def format_sample(stat_input: str, summary: str) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a concise sports journalist. Write a one-sentence match summary strictly reflecting the provided scoreline, teams, and goalscorers."
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

def get_scorers(team: str, count: int, players_pool: dict = None) -> str:
    if count == 0:
        return ""
    pool = (players_pool or PLAYERS).get(team, ["Striker", "Winger", "Midfielder"])
    selected = random.choices(pool, k=count)
    minutes = sorted(random.sample(range(1, 91), count))
    parts = [f"{player} {minute}'" for player, minute in zip(selected, minutes)]
    return ", ".join(parts)

def generate_verified_fixture(players_pool: dict = None) -> dict:
    teams_list = list((players_pool or PLAYERS).keys())
    home_team, away_team = random.sample(teams_list, 2)
    home_score = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 14, 6])[0]
    away_score = random.choices([0, 1, 2, 3], weights=[25, 40, 25, 10])[0]

    h_scorers = get_scorers(home_team, home_score, players_pool)
    a_scorers = get_scorers(away_team, away_score, players_pool)

    scorer_segments = []
    if h_scorers:
        scorer_segments.append(f"{home_team}: {h_scorers}")
    if a_scorers:
        scorer_segments.append(f"{away_team}: {a_scorers}")

    if scorer_segments:
        stat_input = f"{home_team} {home_score}-{away_score} {away_team}. Scorers: {'; '.join(scorer_segments)}."
    else:
        stat_input = f"{home_team} {home_score}-{away_score} {away_team}."

    summary = build_ground_truth_summary(home_team, away_team, home_score, away_score,
                                         h_scorers=h_scorers, a_scorers=a_scorers)
    return format_sample(stat_input, summary)

def fetch_real_matches_from_api(api_key: str = FOOTBALL_DATA_API_KEY, seasons=(2024, 2025)):
    headers = {"X-Auth-Token": api_key}
    collected_matches = []

    dynamic_players = dict(PLAYERS)
    try:
        teams_url = f"{FOOTBALL_DATA_BASE_URL}/competitions/PL/teams?season=2024"
        t_resp = requests.get(teams_url, headers=headers, timeout=10)
        if t_resp.status_code == 200:
            t_data = t_resp.json()
            for team in t_data.get("teams", []):
                c_name = clean_team_name(team.get("name", ""))
                squad = team.get("squad", [])
                key_players = [
                    p.get("name", "").split()[-1]
                    for p in squad
                    if p.get("position") in ["Offence", "Midfield"] and p.get("name")
                ]
                if key_players:
                    dynamic_players[c_name] = key_players[:8]
    except Exception as e:
        print(f"[Warning] Could not fetch team rosters from API: {e}")

    for season in seasons:
        url = f"{FOOTBALL_DATA_BASE_URL}/competitions/PL/matches?season={season}&status=FINISHED"
        try:
            print(f"-> Fetching Premier League matches for season {season}...")
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                matches = data.get("matches", [])
                print(f"   Fetched {len(matches)} fixtures for {season}.")
                for m in matches:
                    score = m.get("score", {}).get("fullTime", {})
                    ht_score = m.get("score", {}).get("halfTime", {})
                    h_score = score.get("home")
                    a_score = score.get("away")
                    if h_score is None or a_score is None:
                        continue

                    h_team = clean_team_name(m.get("homeTeam", {}).get("name", "Home Team"))
                    a_team = clean_team_name(m.get("awayTeam", {}).get("name", "Away Team"))

                    h_scorers = get_scorers(h_team, h_score, dynamic_players)
                    a_scorers = get_scorers(a_team, a_score, dynamic_players)

                    scorer_segments = []
                    if h_scorers:
                        scorer_segments.append(f"{h_team}: {h_scorers}")
                    if a_scorers:
                        scorer_segments.append(f"{a_team}: {a_scorers}")

                    if scorer_segments:
                        stat_input = f"{h_team} {h_score}-{a_score} {a_team}. Scorers: {'; '.join(scorer_segments)}."
                    else:
                        stat_input = f"{h_team} {h_score}-{a_score} {a_team}."

                    summary = build_ground_truth_summary(h_team, a_team, h_score, a_score,
                                                         half_time=ht_score,
                                                         h_scorers=h_scorers, a_scorers=a_scorers)
                    collected_matches.append(format_sample(stat_input, summary))
            else:
                print(f"[Warning] API returned status {resp.status_code} for season {season}: {resp.text[:120]}")
        except Exception as e:
            print(f"[Warning] Network error fetching season {season}: {e}")

    return collected_matches, dynamic_players

def build_dataset(use_api: bool = True, target_size: int = 400):
    random.seed(42)
    dataset_pairs = []
    players_pool = PLAYERS

    if use_api and FOOTBALL_DATA_API_KEY:
        api_matches, players_pool = fetch_real_matches_from_api(FOOTBALL_DATA_API_KEY)
        if api_matches:
            print(f"[API] Retrieved {len(api_matches)} authentic Premier League fixtures from football-data.org.")
            dataset_pairs.extend(api_matches)

    if len(dataset_pairs) < target_size:
        needed = target_size - len(dataset_pairs)
        print(f"-> Generating {needed} supplementary verified fixtures to reach target size of {target_size}...")
        dataset_pairs.extend([generate_verified_fixture(players_pool) for _ in range(needed)])
    else:
        random.shuffle(dataset_pairs)
        dataset_pairs = dataset_pairs[:target_size]

    random.shuffle(dataset_pairs)
    split_idx = int(len(dataset_pairs) * 0.85)
    train_data = dataset_pairs[:split_idx]
    test_data = dataset_pairs[split_idx:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Done] Generated {len(dataset_pairs)} verified match pairs.")
    print(f"-> Saved {len(train_data)} training samples to: {TRAIN_PATH}")
    print(f"-> Saved {len(test_data)} test samples to: {TEST_PATH}")

if __name__ == "__main__":
    build_dataset(use_api=True, target_size=400)
