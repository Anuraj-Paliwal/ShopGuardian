
import joblib
import os
import logging

MODEL_PATH = "models/fraud_model.pkl"

class FraudModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.model_type = "None"
        self.loaded = False
        self._load_model(model_path)

    def _load_model(self, path):
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
                self.model_type = str(type(self.model))
                self.loaded = True
                logging.info(f"✔️ fraud_model.pkl loaded: {self.model_type}")
            except Exception as e:
                logging.exception("⚠️ Failed to load model file.")
        else:
            logging.warning("⚠️ Model file not found at %s", path)

    def predict(self, features: dict) -> float:
        logging.info("⚠️ Using heuristic fallback for fraud detection (model loaded, input schema unknown)")
        return self._heuristic_score(features)

    def _heuristic_score(self, features):
        amount = features.get("amount", 0)
        score = 0.0
        if amount > 2000:
            score += 0.25
        elif amount > 1000:
            score += 0.15
        if features.get("device_trust_score", 1) < 0.4:
            score += 0.3
        if features.get("country_risk", 0) >= 2:
            score += 0.2
        return min(score, 1.0)

