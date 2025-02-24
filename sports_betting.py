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
from config import *  # Import shared constants from config.py

# Constants
PACKAGE_VERSION = "1.0"
DATABASE_FILE = "bets_database.db"
SLEEP_BETWEEN_CHECKS = 1800  # 30 minutes in seconds
COUNTDOWN_INTERVAL = 300  # 5 minutes in seconds
CONFIDENCE_THRESHOLD = 0.55
MIN_BETS_FOR_TRAINING = 10  # Minimum bets required to train
TRAINING_CYCLE_INTERVAL = 5  # Train every 5 cycles
WAIT_AFTER_BET_HOURS = 2  # Hours to wait after bet commencement

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

# Map Odds API sport keys to ESPN slugs for team sports only
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


class SportsBettingBot:
    def __init__(self):
        """Initialize the bot with model, scaler, and state variables, loading the brain immediately."""
        self.database_initialized = False
        self.team_mappings = self._load_team_mappings()
        self.cycle_count = 0  # Track betting cycles for periodic training
        self.training_lock = threading.Lock()  # Lock for thread-safe model updates

        # Load the model and scaler at initialization
        self.load_model_and_scaler()
        if self.model is None or self.scaler is None:
            raise RuntimeError("Failed to load or initialize the betting brain. Exiting...")

    def _load_team_mappings(self):
        """Load team mappings for ESPN stats API for team sports only."""
        mappings = {}
        for sport_key, slug in ESPN_SLUGS.items():
            try:
                response = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{slug}/teams",
                                        timeout=API_TIMEOUT)
                response.raise_for_status()
                teams = response.json()['sports'][0]['leagues'][0]['teams']
                mappings[sport_key] = {team['team']['displayName']: team['team']['id'] for team in teams}
            except Exception as e:
                print(f"Error fetching team mappings for {sport_key}: {e}")
        return mappings

    def install_packages(self):
        """Ensure required packages are installed (placeholder)."""
        print("Checking and installing dependencies...")

    def setup_database(self):
        """Set up the SQLite database for bets and stats."""
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
                    c.execute(
                        "INSERT INTO user_stats (total_bets, total_wins, total_losses, last_updated) VALUES (0, 0, 0, ?)",
                        (datetime.now(pytz.UTC).isoformat(),))
            self.database_initialized = True

    def load_model_and_scaler(self):
        """Load pre-trained model and scaler or initialize with full feature set."""
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            # Check if scaler matches current feature set and model supports binary classification
            if (sorted(self.scaler.feature_names_in_) != sorted(FEATURE_COLUMNS) or
                    len(self.model.classes_) != 2):
                print("Scaler or model incompatible with current features or classes. Reinitializing...")
                raise FileNotFoundError  # Trigger reinitialization
            print("Loaded model and scaler.")
        except (FileNotFoundError, AttributeError):
            print("Model or scaler not found or incompatible. Initializing with binary class feature set...")
            # Use two samples to ensure binary classification
            dummy_X = pd.DataFrame([
                [1, 2, 0.5, 1, 3, 0.6, 0.7, 0.1, 0.2, 0.0, 0.5, 0.5, 1, 0.1],  # Class 0 sample
                [2, 1, 0.6, 2, 4, 0.7, 0.8, 0.2, 0.3, 0.1, 0.6, 0.6, 0, 0.2]  # Class 1 sample
            ], columns=FEATURE_COLUMNS)
            dummy_y = [0, 1]  # Binary outcomes
            self.scaler = StandardScaler().fit(dummy_X)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42).fit(self.scaler.transform(dummy_X),
                                                                                       dummy_y)
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            print("Initialized and saved new model and scaler with binary classes.")

    def get_odds_data(self):
        """Fetch real odds data from The Odds API for DraftKings and FanDuel."""
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
            response = requests.get('https://api.the-odds-api.com/v4/sports/upcoming/odds/', params=params,
                                    timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            bets = []
            for event in data:
                commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).replace(
                    tzinfo=pytz.UTC)
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
        """Fetch real-time team stats from ESPN API for team sports, fallback for others."""
        if sport_key in self.team_mappings and team in self.team_mappings[sport_key]:
            team_id = self.team_mappings[sport_key][team]
            slug = ESPN_SLUGS.get(sport_key, sport_key)
            try:
                response = requests.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/{slug}/teams/{team_id}/statistics",
                    timeout=API_TIMEOUT)
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
                home_win_pct = win_pct  # Placeholder; ideally fetch from API
                away_win_pct = win_pct  # Placeholder; ideally fetch from API
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
        """Extract features for a bet with enhanced real-time stats."""
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
        """Calculate win probability using Log5 method."""
        return (p_a * (1 - p_b)) / (p_a * (1 - p_b) + (1 - p_a) * p_b) if (p_a * (1 - p_b) + (
                    1 - p_a) * p_b) > 0 else 0.5

    def display_bet(self, bet, confidence, rank):
        """Display a betting pick to the user."""
        print(
            f"Pick {rank}: {bet['outcome']} to win ({bet['team1']} vs {bet['team2']}) at {bet['odds']} odds on {bet['bookmaker']} (Confidence: {confidence * 100:.2f}%)")

    def display_countdown(self, remaining_time):
        """Display countdown to next betting cycle every 5 minutes."""
        while remaining_time > 0:
            minutes_left = int(remaining_time // 60)
            print(f"Time until next betting cycle: {minutes_left} minutes")
            time.sleep(min(COUNTDOWN_INTERVAL, remaining_time))
            remaining_time -= COUNTDOWN_INTERVAL
            if remaining_time <= 0:
                print("Starting next betting cycle now...")

    def fetch_game_result(self, sport_key, team1, team2, commence_time):
        """Fetch game result from The Odds API scores endpoint."""
        if not sport_key or sport_key == 'unknown':
            print(f"Missing sport_key for {team1} vs {team2}, skipping result fetch")
            return None
        try:
            params = {'apiKey': ODDS_API_KEY, 'daysFrom': 1}
            response = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/", params=params,
                                    timeout=API_TIMEOUT)
            response.raise_for_status()
            games = response.json()
            for game in games:
                if (game['home_team'] == team1 and game['away_team'] == team2 and
                        game['commence_time'] == commence_time and game['completed']):
                    scores_home = float(game['scores'][0]['score']) if game['scores'][0]['name'] == team1 else float(
                        game['scores'][1]['score'])
                    scores_away = float(game['scores'][1]['score']) if game['scores'][1]['name'] == team2 else float(
                        game['scores'][0]['score'])
                    return 1 if scores_home > scores_away else 0
            print(f"No completed result yet for {team1} vs {team2}")
            return None
        except Exception as e:
            print(f"Error fetching result for {team1} vs {team2}: {e}")
            return None

    def update_user_stats(self, result):
        """Update user stats with the latest result."""
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT total_bets, total_wins, total_losses FROM user_stats WHERE id = 1")
            row = c.fetchone()
            if row:
                total_bets, total_wins, total_losses = row
                total_bets += 1
                total_wins += 1 if result == 1 else 0
                total_losses += 1 if result == 0 else 0
                c.execute(
                    "UPDATE user_stats SET total_bets = ?, total_wins = ?, total_losses = ?, last_updated = ? WHERE id = 1",
                    (total_bets, total_wins, total_losses, datetime.now(pytz.UTC).isoformat()))
            conn.commit()

    def check_unresolved_bets(self):
        """Check and resolve past bets with missing results."""
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT id, sport, team1, team2, commence_time FROM bets WHERE result IS NULL")
            unresolved_bets = c.fetchall()
            if unresolved_bets:
                print(f"Checking {len(unresolved_bets)} unresolved bets...")
                for bet_id, sport_key, team1, team2, commence_time in unresolved_bets:
                    result = self.fetch_game_result(sport_key, team1, team2, commence_time)
                    if result is not None:
                        c.execute("UPDATE bets SET result = ? WHERE id = ?", (result, bet_id))
                        self.update_user_stats(result)
                        print(f"Updated result for {team1} vs {team2}: {'Won' if result == 1 else 'Lost'}")
            conn.commit()

    def train_model(self):
        """Train the model with existing bet data in a separate thread."""
        with self.training_lock:
            try:
                with sqlite3.connect(DATABASE_FILE) as conn:
                    existing_df = pd.read_sql_query("SELECT * FROM bets WHERE result IS NOT NULL", conn)

                if len(existing_df) < MIN_BETS_FOR_TRAINING:
                    print(f"Not enough data to train: {len(existing_df)} bets found, need {MIN_BETS_FOR_TRAINING}")
                    return False

                X = existing_df[FEATURE_COLUMNS].fillna({
                    'odds': 0, 'opposing_odds': 0, 'win_pct': 0.5, 'net_ppg': 0,
                    'recent_wins': 0.5, 'betting_pct': 0.5, 'win_prob': 0.5,
                    'win_pct_diff': 0, 'net_ppg_diff': 0, 'momentum': 0,
                    'home_win_pct': 0.5, 'away_win_pct': 0.5, 'home_advantage': 0,
                    'ev': 0
                })
                y = existing_df['result']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced')
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained with {len(existing_df)} bets. Test accuracy: {accuracy:.2f}")

                self.model = model
                self.scaler = scaler
                joblib.dump(self.model, MODEL_FILE)
                joblib.dump(self.scaler, SCALER_FILE)
                print("Model and scaler updated and saved successfully")
                return True

            except Exception as e:
                print(f"Error training model: {e}")
                return False

    def train_in_background(self):
        """Run training in a separate thread."""
        threading.Thread(target=self.train_model, daemon=True).start()

    def display_stats(self):
        """Display bot performance stats from the database."""
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT total_bets, total_wins, total_losses, last_updated FROM user_stats WHERE id = 1")
            row = c.fetchone()
        if row:
            total_bets, total_wins, total_losses, last_updated = row
            accuracy = total_wins / total_bets * 100 if total_bets > 0 else 0
            print(f"\n--- Your Betting Stats ---")
            print(f"Total Bets: {total_bets}")
            print(f"Wins: {total_wins}")
            print(f"Losses: {total_losses}")
            print(f"Accuracy: {accuracy:.0f}%")
            print(f"Last Updated: {last_updated[:19]}")
            print("-------------------------\n")
        else:
            print("No stats available yet.")

    def run(self):
        """Main method to run the betting bot with smarter logic and training."""
        self.install_packages()
        print(f"Welcome to Sports Betting Bot v{PACKAGE_VERSION}!")

        self.check_unresolved_bets()
        self.setup_database()
        self.display_stats()

        try:
            while True:
                try:
                    num_picks = int(input("How many picks do you want (1 or more)? "))
                    if num_picks > 0:
                        break
                except ValueError:
                    print("Please enter a valid number greater than 0.")
        except KeyboardInterrupt:
            print("Interrupted by user during input. Returning to main menu...")
            return

        paused = False
        while True:
            try:
                if not paused:
                    odds_data = self.get_odds_data()
                    if not odds_data:
                        print("No games to bet on right now. Checking again in 30 minutes...")
                        self.display_countdown(SLEEP_BETWEEN_CHECKS)
                        continue

                    print(f"Checking {len(odds_data)} upcoming games for {num_picks} picks...")

                    all_bets = []
                    for bet in odds_data:
                        features = self.evaluate_bet(bet, bet.get('opposing_odds'))
                        game_outcome = (features['team1'], features['team2'], features['outcome'])
                        X = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                        X_scaled = self.scaler.transform(X)
                        confidence = self.model.predict_proba(X_scaled)[0][1]  # Index 1 for positive class (win)
                        weighted_score = (confidence * 0.7) + (features['ev'] * 0.3) if features[
                                                                                            'ev'] > 0 else confidence
                        all_bets.append((bet, confidence, weighted_score, game_outcome))

                    all_bets.sort(key=lambda x: x[2], reverse=True)
                    selected_game_outcomes = set()
                    top_bets = []

                    for bet, confidence, weighted_score, game_outcome in all_bets:
                        if game_outcome not in selected_game_outcomes and confidence >= CONFIDENCE_THRESHOLD:
                            selected_game_outcomes.add(game_outcome)
                            top_bets.append((bet, confidence))
                        if len(top_bets) == num_picks:
                            break

                    if len(top_bets) < num_picks:
                        print(
                            f"Only {len(top_bets)} unique picks available with confidence >= {CONFIDENCE_THRESHOLD * 100:.0f}%.")

                    bet_ids = []
                    with sqlite3.connect(DATABASE_FILE) as conn:
                        c = conn.cursor()
                        for i, (bet, confidence) in enumerate(top_bets, 1):
                            self.display_bet(bet, confidence, i)
                            features = self.evaluate_bet(bet, bet.get('opposing_odds'))
                            c.execute('''INSERT INTO bets (sport, team1, team2, outcome, odds, opposing_odds, win_pct, net_ppg, 
                                        recent_wins, betting_pct, win_prob, home_team, win_pct_diff, net_ppg_diff, momentum, 
                                        home_win_pct, away_win_pct, home_advantage)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                      (features['sport'], features['team1'], features['team2'], features['outcome'],
                                       features['odds'], features['opposing_odds'], features['win_pct'],
                                       features['net_ppg'],
                                       features['recent_wins'], features['betting_pct'], features['win_prob'],
                                       features['team1'], features['win_pct_diff'], features['net_ppg_diff'],
                                       features['momentum'], features['home_win_pct'], features['away_win_pct'],
                                       features['home_advantage']))
                            bet_ids.append((c.lastrowid, features))
                        conn.commit()

                    for bet_id, features in bet_ids:
                        commence_time_dt = datetime.fromisoformat(
                            features['commence_time'].replace('Z', '+00:00')).replace(tzinfo=pytz.UTC)
                        wait_until = commence_time_dt + timedelta(hours=WAIT_AFTER_BET_HOURS)
                        if datetime.now(pytz.UTC) > wait_until:
                            result = self.fetch_game_result(features['sport'], features['team1'], features['team2'],
                                                            features['commence_time'])
                            if result is not None:
                                with sqlite3.connect(DATABASE_FILE) as conn:
                                    c = conn.cursor()
                                    c.execute("UPDATE bets SET result = ? WHERE id = ?", (result, bet_id))
                                    self.update_user_stats(result)
                                    conn.commit()
                                print(
                                    f"Result for {features['team1']} vs {features['team2']}: {'Won' if result == 1 else 'Lost'}")

                    print("Waiting for next betting cycle...")
                    self.display_countdown(SLEEP_BETWEEN_CHECKS)

                    self.cycle_count += 1
                    if self.cycle_count % TRAINING_CYCLE_INTERVAL == 0:
                        print("Training model in background...")
                        self.train_in_background()

                print("Commands: pause, resume, train, exit")
                command = input("Enter command: ").strip().lower()

                if command == "pause":
                    paused = True
                    print(
                        "Sports betting paused. Enter 'resume' to continue, 'train' to train model, or 'exit' to return to menu.")
                elif command == "resume":
                    paused = False
                    print("Resuming sports betting...")
                elif command == "train":
                    print("Training model in background...")
                    self.train_in_background()
                elif command == "exit":
                    print("Exiting sports betting and returning to main menu...")
                    self.check_unresolved_bets()
                    return  # Exit to main menu
                else:
                    print("Invalid command. Use: pause, resume, train, exit")

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting gracefully...")
                self.check_unresolved_bets()
                return  # Exit to main menu


if __name__ == "__main__":
    bot = SportsBettingBot()
    bot.run()