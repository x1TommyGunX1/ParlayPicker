import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
from datetime import datetime, timedelta
import pytz
import requests
import joblib
import threading
from flask import Flask, jsonify, request
from config import *  # Import shared constants from config.py

# Constants
PACKAGE_VERSION = "1.0"
DATABASE_FILE = "bets_database.db"
SLEEP_BETWEEN_CHECKS = 1800  # 30 minutes in seconds
COUNTDOWN_INTERVAL = 300  # 5 minutes in seconds
CONFIDENCE_THRESHOLD = 0.55
MIN_BETS_FOR_TRAINING = 10
TRAINING_CYCLE_INTERVAL = 5
WAIT_AFTER_BET_HOURS = 2

# Define feature list globally
FEATURE_COLUMNS = [
    'odds', 'opposing_odds', 'win_pct', 'net_ppg', 'recent_wins',
    'betting_pct', 'win_prob', 'win_pct_diff', 'net_ppg_diff', 'momentum',
    'home_win_pct', 'away_win_pct', 'home_advantage', 'ev'
]

# Expanded sport names mapping
SPORT_NAMES = {
    'baseball_mlb': 'MLB Baseball',
    'basketball_nba': 'NBA Basketball',
    'basketball_wnba': 'WNBA Basketball',
    'football_nfl': 'NFL Football',
    'football_ncaaf': 'NCAA Football',
    'hockey_nhl': 'NHL Hockey',
    'soccer_usa_mls': 'MLS Soccer',
    'soccer_eng_1': 'Premier League Soccer',
    'soccer_esp_1': 'La Liga Soccer',
    'soccer_ita_1': 'Serie A Soccer',
    'tennis_atp': 'ATP Tennis',
    'tennis_wta': 'WTA Tennis',
    'mma_mixed_martial_arts': 'MMA',
    'boxing_boxing': 'Boxing',
    'golf_pga': 'PGA Golf'
}

# Map Odds API sport keys to ESPN slugs
ESPN_SLUGS = {
    'baseball_mlb': 'baseball/mlb',
    'basketball_nba': 'basketball/nba',
    'basketball_wnba': 'basketball/wnba',
    'football_nfl': 'football/nfl',
    'football_ncaaf': 'football/college-football',
    'hockey_nhl': 'hockey/nhl',
    'soccer_usa_mls': 'soccer/usa.1',
    'soccer_eng_1': 'soccer/eng.1',
    'soccer_esp_1': 'soccer/esp.1',
    'soccer_ita_1': 'soccer/ita.1'
}

# Default stats for fallback
DEFAULT_STATS = {
    'win_pct': 0.5,
    'net_ppg': 0.0,
    'recent_wins': 0.5,
    'momentum': 0.0,
    'home_win_pct': 0.5,
    'away_win_pct': 0.5
}

# Initialize Flask app
app = Flask(__name__)

class SportsBettingBot:
    def __init__(self):
        """Initialize the bot with model, scaler, and state variables."""
        self.database_initialized = False
        self.team_mappings = self._load_team_mappings()
        self.cycle_count = 0
        self.training_lock = threading.Lock()
        self.load_model_and_scaler()
        if self.model is None or self.scaler is None:
            raise RuntimeError("Failed to load or initialize the betting brain.")

    def _load_team_mappings(self):
        mappings = {}
        for sport_key, slug in ESPN_SLUGS.items():
            try:
                response = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{slug}/teams", timeout=API_TIMEOUT)
                response.raise_for_status()
                teams = response.json()['sports'][0]['leagues'][0]['teams']
                mappings[sport_key] = {team['team']['displayName']: team['team']['id'] for team in teams}
            except Exception as e:
                print(f"Error fetching team mappings for {sport_key}: {e}")
        return mappings

    def setup_database(self):
        if not self.database_initialized:
            print("Setting up database...")
            with sqlite3.connect(DATABASE_FILE) as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT, team1 TEXT, team2 TEXT, outcome TEXT,
                    odds REAL, opposing_odds REAL, win_pct REAL, net_ppg REAL, recent_wins REAL,
                    betting_pct REAL, win_prob REAL, home_team TEXT, win_pct_diff REAL, net_ppg_diff REAL,
                    momentum REAL, home_win_pct REAL, away_win_pct REAL,
                    result INTEGER, timestamp TEXT, commence_time TEXT, bookmaker TEXT, home_advantage INTEGER)''')
                c.execute('''CREATE TABLE IF NOT EXISTS user_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, total_bets INTEGER DEFAULT 0,
                    total_wins INTEGER DEFAULT 0, total_losses INTEGER DEFAULT 0, last_updated TEXT)''')
                c.execute("SELECT COUNT(*) FROM user_stats")
                if c.fetchone()[0] == 0:
                    c.execute("INSERT INTO user_stats (total_bets, total_wins, total_losses, last_updated) VALUES (0, 0, 0, ?)",
                              (datetime.now(pytz.UTC).isoformat(),))
            self.database_initialized = True

    def load_model_and_scaler(self):
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            if (sorted(self.scaler.feature_names_in_) != sorted(FEATURE_COLUMNS) or
                len(self.model.classes_) != 2):
                print("Scaler or model incompatible. Reinitializing...")
                raise FileNotFoundError
            print("Loaded model and scaler.")
        except (FileNotFoundError, AttributeError):
            print("Initializing with binary class feature set...")
            dummy_X = pd.DataFrame([
                [1, 2, 0.5, 1, 3, 0.6, 0.7, 0.1, 0.2, 0.0, 0.5, 0.5, 1, 0.1],
                [2, 1, 0.6, 2, 4, 0.7, 0.8, 0.2, 0.3, 0.1, 0.6, 0.6, 0, 0.2]
            ], columns=FEATURE_COLUMNS)
            dummy_y = [0, 1]
            self.scaler = StandardScaler().fit(dummy_X)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42).fit(self.scaler.transform(dummy_X), dummy_y)
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            print("Initialized and saved new model and scaler.")

    def get_odds_data(self):
        try:
            today = datetime.now(pytz.UTC).date()
            tomorrow = today + timedelta(days=1)
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': 'h2h',
                'oddsFormat': 'decimal',
                'bookmakers': 'draftkings,fanduel',
                'dateFrom': today.isoformat(),
                'dateTo': tomorrow.isoformat()
            }
            response = requests.get('https://api.the-odds-api.com/v4/sports/upcoming/odds/', params=params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            bets = []
            for event in data:
                commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).replace(tzinfo=pytz.UTC)
                if commence_time.date() not in [today, tomorrow] or not event['bookmakers']:
                    continue
                sport_key = event.get('sport_key', 'unknown')
                for bookmaker in event['bookmakers']:
                    if bookmaker['key'] in ['draftkings', 'fanduel']:
                        market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
                        if market and len(market['outcomes']) >= 2:
                            for outcome in market['outcomes']:
                                opposing_outcome = next(o for o in market['outcomes'] if o['name'] != outcome['name'])
                                bets.append({
                                    'sport': sport_key,
                                    'team1': event['home_team'],
                                    'team2': event['away_team'],
                                    'odds': outcome['price'],
                                    'opposing_odds': opposing_outcome['price'],
                                    'outcome': outcome['name'],
                                    'bookmaker': bookmaker['key'],
                                    'commence_time': event['commence_time']
                                })
            print(f"Retrieved {len(bets)} betting options from DraftKings and FanDuel.")
            return bets
        except Exception as e:
            print(f"Error fetching odds data: {e}")
            return []

    def scrape_team_stats(self, team, sport_key):
        if sport_key in self.team_mappings and team in self.team_mappings[sport_key]:
            team_id = self.team_mappings[sport_key][team]
            slug = ESPN_SLUGS.get(sport_key, sport_key)
            try:
                response = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{slug}/teams/{team_id}/statistics", timeout=API_TIMEOUT)
                response.raise_for_status()
                stats = response.json().get('stats', {})
                wins = int(stats.get('wins', 0))
                losses = int(stats.get('losses', 0))
                games = wins + losses
                win_pct = wins / games if games > 0 else DEFAULT_STATS['win_pct']
                ppg = float(stats.get('points', 0)) / games if games > 0 else DEFAULT_STATS['net_ppg']
                opp_ppg = float(stats.get('pointsAgainst', 0)) / games if games > 0 else DEFAULT_STATS['net_ppg']
                net_ppg = ppg - opp_ppg
                recent_wins = min(wins, 5) / 5 if games >= 5 else win_pct
                momentum = (recent_wins - win_pct) if games >= 5 else 0.0
                home_win_pct = win_pct
                away_win_pct = win_pct
                return {
                    'win_pct': win_pct,
                    'net_ppg': net_ppg,
                    'recent_wins': recent_wins,
                    'momentum': momentum,
                    'home_win_pct': home_win_pct,
                    'away_win_pct': away_win_pct
                }
            except Exception as e:
                print(f"Error fetching stats for {team} in {sport_key}: {e}")
        return DEFAULT_STATS

    def evaluate_bet(self, bet, opposing_odds):
        team_stats = self.scrape_team_stats(bet['outcome'], bet['sport'])
        opponent = bet['team2'] if bet['outcome'] == bet['team1'] else bet['team1']
        opponent_stats = self.scrape_team_stats(opponent, bet['sport'])

        win_prob = self.log5_probability(team_stats['win_pct'], opponent_stats['win_pct'])
        betting_pct = bet['odds'] / (bet['odds'] + opposing_odds) if (bet['odds'] + opposing_odds) > 0 else 0.5
        home_advantage = 1 if bet['outcome'] == bet['team1'] else 0
        implied_prob = 1 / bet['odds']
        ev = (win_prob * (bet['odds'] - 1)) - (1 - win_prob)

        features = {
            'sport': bet['sport'],
            'team1': bet['team1'],
            'team2': bet['team2'],
            'outcome': bet['outcome'],
            'odds': bet['odds'],
            'opposing_odds': opposing_odds if opposing_odds else bet['odds'] * 0.95,
            'win_pct': team_stats['win_pct'],
            'net_ppg': team_stats['net_ppg'],
            'recent_wins': team_stats['recent_wins'],
            'betting_pct': betting_pct,
            'win_prob': win_prob,
            'win_pct_diff': team_stats['win_pct'] - opponent_stats['win_pct'],
            'net_ppg_diff': team_stats['net_ppg'] - opponent_stats['net_ppg'],
            'momentum': team_stats['momentum'],
            'home_win_pct': team_stats['home_win_pct'] if home_advantage else team_stats['away_win_pct'],
            'away_win_pct': opponent_stats['away_win_pct'] if home_advantage else opponent_stats['home_win_pct'],
            'home_advantage': home_advantage,
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'commence_time': bet['commence_time'],
            'bookmaker': bet['bookmaker'],
            'ev': ev
        }
        return features

    def log5_probability(self, p_a, p_b):
        return (p_a * (1 - p_b)) / (p_a * (1 - p_b) + (1 - p_a) * p_b) if (p_a * (1 - p_b) + (1 - p_a) * p_b) > 0 else 0.5

    def get_picks(self, num_picks):
        odds_data = self.get_odds_data()
        if not odds_data:
            return []

        all_bets = []
        for bet in odds_data:
            features = self.evaluate_bet(bet, bet.get('opposing_odds'))
            game_outcome = (features['team1'], features['team2'], features['outcome'])
            X = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            X_scaled = self.scaler.transform(X)
            confidence = self.model.predict_proba(X_scaled)[0][1]
            weighted_score = (confidence * 0.7) + (features['ev'] * 0.3) if features['ev'] > 0 else confidence
            all_bets.append((bet, confidence, weighted_score, game_outcome))

        all_bets.sort(key=lambda x: x[2], reverse=True)
        selected_game_outcomes = set()
        top_bets = []

        for bet, confidence, weighted_score, game_outcome in all_bets:
            if game_outcome not in selected_game_outcomes and confidence >= CONFIDENCE_THRESHOLD:
                selected_game_outcomes.add(game_outcome)
                top_bets.append({
                    'rank': len(top_bets) + 1,
                    'outcome': bet['outcome'],
                    'team1': bet['team1'],
                    'team2': bet['team2'],
                    'odds': bet['odds'],
                    'bookmaker': bet['bookmaker'],
                    'confidence': confidence * 100
                })
            if len(top_bets) == num_picks:
                break

        return top_bets

# Flask Routes
@app.route('/')
def home():
    return jsonify({"message": f"Welcome to Sports Betting Bot v{PACKAGE_VERSION}! Use /picks to get betting picks."})

@app.route('/picks', methods=['GET'])
def get_picks():
    num_picks = int(request.args.get('num_picks', 2))
    bot = SportsBettingBot()  # Create bot instance
    bot.setup_database()      # Ensure database is ready
    picks = bot.get_picks(num_picks)
    return jsonify({"picks": picks})

if __name__ == "__main__":
    bot = SportsBettingBot()  # Test locally
    bot.run()
    app.run(debug=True, host='0.0.0.0', port=5000)
