import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging
from config import *  # Import shared constants

# Logging setup
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    filename=LOG_FILE,
    filemode=LOG_FILE_MODE
)
logger = logging.getLogger(__name__)


def train_with_existing_data():
    """Train the RandomForestClassifier with existing bet data from the database."""
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            existing_df = pd.read_sql_query("SELECT * FROM bets WHERE result IS NOT NULL", conn)

        if len(existing_df) < MIN_BETS_FOR_TRAINING:
            logger.warning(f"Not enough data to train: {len(existing_df)} bets found, need {MIN_BETS_FOR_TRAINING}")
            print(f"Need at least {MIN_BETS_FOR_TRAINING} bets to train! Found {len(existing_df)}.")
            return False

        feature_cols = [
            'odds', 'opposing_odds', 'win_pct', 'net_ppg', 'recent_wins',
            'betting_pct', 'win_prob', 'win_pct_diff', 'net_ppg_diff', 'home_advantage'
        ]

        X = existing_df[feature_cols] if all(col in existing_df.columns for col in feature_cols) else \
            existing_df.reindex(columns=feature_cols, fill_value=0)
        X = X.fillna({
            'odds': 0, 'opposing_odds': 0, 'win_pct': 0.5, 'net_ppg': 0,
            'recent_wins': 0.5, 'betting_pct': 0.5, 'win_prob': 0.5,
            'win_pct_diff': 0, 'net_ppg_diff': 0, 'home_advantage': 0
        })
        y = existing_df['result']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with {len(existing_df)} bets. Test accuracy: {accuracy:.2f}")
        print(f"Trained model with {len(existing_df)} bets. Test accuracy: {accuracy:.2f}")

        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Model and scaler saved successfully")
        print("Model and scaler saved successfully")
        return True

    except Exception as e:
        logger.error(f"Error training model: {e}")
        print(f"Error training model: {e}")
        return False


def main():
    """Main function to run the trainer."""
    success = train_with_existing_data()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed. Check the log for details.")


if __name__ == "__main__":
    main()