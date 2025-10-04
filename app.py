from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid
import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production!

# File paths
TRANSACTION_FILE = "transactions.json"
ALERTS_FILE = "alerts.log"
USERS_FILE = "users.json"
MODEL_FILE = "fraud_model.pkl"
FEATURES_FILE = "feature_names.json"

# System start time for calculating transaction times
SYSTEM_START_TIME = datetime.utcnow()

# Create files if they don't exist
for file in [TRANSACTION_FILE, ALERTS_FILE]:
    if not os.path.exists(file):
        open(file, 'w').close()

# Initialize users file with default users
if not os.path.exists(USERS_FILE):
    default_users = {
        "admin": {"password": "admin123", "role": "admin"},
        "user1": {"password": "user123", "role": "user"},
        "demo": {"password": "demo123", "role": "user"}
    }
    with open(USERS_FILE, 'w') as f:
        json.dump(default_users, f, indent=2)

# Load ML model and feature names
try:
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, 'r') as f:
        feature_names = json.load(f)
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML model: {e}")
    model = None
    feature_names = None

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- ML Prediction Function ---
def predict_fraud(transaction_data):
    """Predict if a transaction is fraudulent using the trained model"""
    if model is None or feature_names is None:
        # Fallback: simple rule-based if model not loaded
        amount = float(transaction_data.get('Amount', 0))
        if amount > 2000:
            return {"label": "fraudulent", "fraud_probability": 0.75}
        return {"label": "genuine", "fraud_probability": 0.15}
    
    try:
        # Create DataFrame with all required features
        features_dict = {}
        for feature in feature_names:
            if feature in transaction_data:
                features_dict[feature] = transaction_data[feature]
            elif feature.startswith('V'):
                # Use provided V features or generate neutral ones
                features_dict[feature] = transaction_data.get(feature, 0.0)
            elif feature == 'Time':
                features_dict[feature] = transaction_data.get('Time', 0)
            elif feature == 'Amount':
                features_dict[feature] = transaction_data.get('Amount', 0)
            else:
                features_dict[feature] = 0
        
        df = pd.DataFrame([features_dict])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return {
            "label": "fraudulent" if prediction == 1 else "genuine",
            "fraud_probability": float(probability[1])
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback prediction
        amount = float(transaction_data.get('Amount', 0))
        if amount > 2000:
            return {"label": "fraudulent", "fraud_probability": 0.75}
        return {"label": "genuine", "fraud_probability": 0.15}

# --- Detect suspicious transactions ---
def detect_suspicious(transaction):
    alerts = []
    if float(transaction.get("Amount", 0)) > 1000:
        alerts.append("High-value transaction")
    if transaction.get("label") == "fraudulent":
        alerts.append("ML Model: Fraudulent pattern detected")
    return alerts

# --- Log transaction ---
def log_transaction(transaction):
    transaction["transaction_id"] = str(uuid.uuid4())
    transaction["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    # Auto-calculate Time if not provided (seconds since system start)
    if "Time" not in transaction:
        elapsed = (datetime.utcnow() - SYSTEM_START_TIME).total_seconds()
        transaction["Time"] = round(elapsed, 2)
    
    # Get ML prediction
    prediction = predict_fraud(transaction)
    transaction["label"] = prediction["label"]
    transaction["fraud_probability"] = prediction["fraud_probability"]
    
    # Add user who created the transaction
    transaction["created_by"] = session.get('username', 'system')
    
    # Save transaction
    with open(TRANSACTION_FILE, "a") as f:
        f.write(json.dumps(transaction) + "\n")
    
    # Check alerts
    alerts = detect_suspicious(transaction)
    if alerts:
        alert_message = (
            f"{datetime.utcnow().isoformat()}Z | "
            f"Transaction {transaction['transaction_id']} | "
            f"User: {transaction.get('user_id', 'N/A')} | "
            f"Amount: {transaction.get('Amount', 'N/A')} | "
            f"Fraud Probability: {prediction['fraud_probability']:.2%} | "
            f"Alerts: {', '.join(alerts)}\n"
        )
        print(f"⚠️ Fraud Alert: {alert_message.strip()}")
        with open(ALERTS_FILE, "a") as f:
            f.write(alert_message)
    
    return transaction

# --- Routes ---

@app.route("/")
def index():
    if 'username' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('transaction_page'))
        return redirect(url_for('dashboard'))
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        # Load users
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users and users[username]["password"] == password:
            session['username'] = username
            session['role'] = users[username]["role"]
            
            if users[username]["role"] == "admin":
                return redirect(url_for('transaction_page'))
            return redirect(url_for('dashboard'))
        
        return render_template("login.html", error="Invalid username or password")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("transactions.html", username=session['username'])

@app.route("/transactions")
@admin_required
def transaction_page():
    return render_template("dashboard.html", username=session['username'])

# --- API Endpoints ---

@app.route("/api/transaction", methods=["POST"])
@login_required
def create_transaction():
    data = request.json
    if not data:
        return jsonify({"error": "No transaction data provided"}), 400
    
    # Add default user_id if not provided
    if "user_id" not in data:
        data["user_id"] = session['username']
    
    logged_tx = log_transaction(data)
    return jsonify({
        "status": "success",
        "transaction_id": logged_tx["transaction_id"],
        "label": logged_tx["label"],
        "fraud_probability": logged_tx["fraud_probability"]
    }), 200

@app.route("/api/transactions")
@login_required
def get_transactions():
    transactions = []
    try:
        with open(TRANSACTION_FILE) as f:
            for line in f:
                if line.strip():
                    transactions.append(json.loads(line))
    except FileNotFoundError:
        pass
    
    # If user is not admin, filter to only their transactions
    if session.get('role') != 'admin':
        username = session['username']
        transactions = [t for t in transactions if t.get('user_id') == username or t.get('created_by') == username]
    
    # Return most recent first
    return jsonify(list(reversed(transactions[-100:])))

@app.route("/api/alerts")
@admin_required
def get_alerts():
    alerts = []
    try:
        with open(ALERTS_FILE) as f:
            alerts = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        pass
    return jsonify(list(reversed(alerts[-50:])))

@app.route("/api/stats")
@login_required
def get_stats():
    transactions = []
    try:
        with open(TRANSACTION_FILE) as f:
            for line in f:
                if line.strip():
                    transactions.append(json.loads(line))
    except FileNotFoundError:
        pass
    
    # Filter for non-admin users
    if session.get('role') != 'admin':
        username = session['username']
        transactions = [t for t in transactions if t.get('user_id') == username or t.get('created_by') == username]
    
    total = len(transactions)
    fraudulent = sum(1 for t in transactions if t.get('label') == 'fraudulent')
    genuine = total - fraudulent
    total_amount = sum(float(t.get('Amount', 0)) for t in transactions)
    
    return jsonify({
        "total": total,
        "fraudulent": fraudulent,
        "genuine": genuine,
        "total_amount": round(total_amount, 2),
        "fraud_rate": round((fraudulent / total * 100) if total > 0 else 0, 2)
    })

# --- Simulator Control Endpoints ---

@app.route("/api/simulator/start", methods=["POST"])
@admin_required
def start_simulator():
    global simulator
    
    data = request.json or {}
    duration = data.get('duration_minutes')  # None = infinite
    rate = data.get('transactions_per_minute', 5)
    
@app.route("/api/simulator/stop", methods=["POST"])
@admin_required
def stop_simulator():
    global simulator
    
    if simulator and simulator.stop():
        return jsonify({"status": "success", "message": "Simulator stopped"}), 200
    else:
        return jsonify({"status": "error", "message": "Simulator not running"}), 400

@app.route("/api/simulator/status")
@admin_required
def simulator_status():
    global simulator
    
    if simulator:
        status = simulator.status()
        return jsonify(status), 200
    else:
        return jsonify({"is_running": False, "transaction_count": 0}), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid, json, os, threading, time
from datetime import datetime
import joblib
import pandas as pd
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# -----------------------------
# Files & Model Setup
# -----------------------------
TRANSACTION_FILE = "transactions.json"
ALERTS_FILE = "alerts.log"
USERS_FILE = "users.json"
MODEL_FILE = "fraud_model.pkl"
FEATURES_FILE = "feature_names.json"

SYSTEM_START_TIME = datetime.utcnow()

# Load model
try:
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, 'r') as f:
        feature_names = json.load(f)
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML model: {e}")
    model = None
    feature_names = None

# Ensure files exist
for file in [TRANSACTION_FILE, ALERTS_FILE]:
    if not os.path.exists(file):
        open(file, 'w').close()

# Default users
if not os.path.exists(USERS_FILE):
    default_users = {
        "admin": {"password": "admin123", "role": "admin"},
        "user1": {"password": "user123", "role": "user"},
        "demo": {"password": "demo123", "role": "user"}
    }
    with open(USERS_FILE, 'w') as f:
        json.dump(default_users, f, indent=2)

# -----------------------------
# Authentication
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# -----------------------------
# ML Prediction
# -----------------------------
def predict_fraud(tx_data):
    if model is None or feature_names is None:
        amount = float(tx_data.get('Amount', 0))
        return {"label": "fraudulent" if amount > 2000 else "genuine",
                "fraud_probability": 0.75 if amount > 2000 else 0.15}
    try:
        features_dict = {f: tx_data.get(f, 0.0) for f in feature_names}
        df = pd.DataFrame([features_dict])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return {"label": "fraudulent" if pred == 1 else "genuine",
                "fraud_probability": float(prob)}
    except:
        amount = float(tx_data.get('Amount', 0))
        return {"label": "fraudulent" if amount > 2000 else "genuine",
                "fraud_probability": 0.75 if amount > 2000 else 0.15}

# -----------------------------
# Transaction Handling
# -----------------------------
def log_transaction(transaction, created_by=None):
    transaction["transaction_id"] = str(uuid.uuid4())
    transaction["timestamp"] = datetime.utcnow().isoformat() + "Z"

    if "Time" not in transaction:
        elapsed = (datetime.utcnow() - SYSTEM_START_TIME).total_seconds()
        transaction["Time"] = round(elapsed, 2)

    prediction = predict_fraud(transaction)
    transaction["label"] = prediction["label"]
    transaction["fraud_probability"] = prediction["fraud_probability"]

    transaction["created_by"] = created_by or session.get('username', 'system')

    with open(TRANSACTION_FILE, "a") as f:
        f.write(json.dumps(transaction) + "\n")

    alerts = detect_suspicious(transaction)
    if alerts:
        alert_message = (
            f"{datetime.utcnow().isoformat()}Z | "
            f"Transaction {transaction['transaction_id']} | "
            f"User: {transaction.get('user_id', 'N/A')} | "
            f"Amount: {transaction.get('Amount', 'N/A')} | "
            f"Fraud Probability: {prediction['fraud_probability']:.2%} | "
            f"Alerts: {', '.join(alerts)}\n"
        )
        print(f"⚠️ Fraud Alert: {alert_message.strip()}")
        with open(ALERTS_FILE, "a") as f:
            f.write(alert_message)

    return transaction

def annotate_transactions_with_model(transactions):
    if model is None:
        return transactions
    annotated = []
    for tx in transactions:
        try:
            feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            row_dict = {f: tx.get(f, 0.0) for f in feature_order}
            df = pd.DataFrame([row_dict])
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None
            tx["model_label"] = "fraudulent" if pred == 1 else "genuine"
            tx["model_probability"] = float(prob) if prob is not None else None
        except:
            tx["model_label"] = tx.get("label", "genuine")
            tx["model_probability"] = tx.get("fraud_probability", 0.0)
        annotated.append(tx)
    return annotated

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    if 'username' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('transaction_page'))
        return redirect(url_for('dashboard'))
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        if username in users and users[username]["password"] == password:
            session['username'] = username
            session['role'] = users[username]["role"]
            return redirect(url_for('transaction_page') if users[username]["role"]=="admin" else url_for('dashboard'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("transactions.html", username=session['username'])

@app.route("/transactions")
@admin_required
def transaction_page():
    return render_template("dashboard.html", username=session['username'])

# -----------------------------
# API
# -----------------------------
@app.route("/api/transaction", methods=["POST"])
@login_required
def create_transaction():
    data = request.json
    if not data:
        return jsonify({"error": "No transaction data provided"}), 400
    if "user_id" not in data:
        data["user_id"] = session['username']
    logged_tx = log_transaction(data)
    return jsonify(logged_tx)

@app.route("/api/transactions")
@login_required
def get_transactions():
    transactions = []
    try:
        with open(TRANSACTION_FILE) as f:
            for line in f:
                if line.strip():
                    transactions.append(json.loads(line))
    except FileNotFoundError:
        pass
    if session.get('role') != 'admin':
        username = session['username']
        transactions = [t for t in transactions if t.get('user_id') == username or t.get('created_by') == username]
    transactions = annotate_transactions_with_model(transactions)
    return jsonify(list(reversed(transactions[-100:])))

@app.route("/api/stats")
@login_required
def get_stats():
    transactions = []
    try:
        with open(TRANSACTION_FILE) as f:
            for line in f:
                if line.strip():
                    transactions.append(json.loads(line))
    except:
        pass
    if session.get('role') != 'admin':
        username = session['username']
        transactions = [t for t in transactions if t.get('user_id') == username or t.get('created_by') == username]
    total = len(transactions)
    fraudulent = sum(1 for t in transactions if t.get('label') == 'fraudulent')
    genuine = total - fraudulent
    total_amount = sum(float(t.get('Amount', 0)) for t in transactions)
    return jsonify({
        "total": total,
        "fraudulent": fraudulent,
        "genuine": genuine,
        "total_amount": round(total_amount,2),
        "fraud_rate": round((fraudulent/total*100) if total>0 else 0,2)
    })

# -----------------------------
# Simulation Thread
# -----------------------------
def simulation_worker():
    while True:
        dummy_tx = {
            "Amount": round(100 + 900 * np.random.rand(), 2),
            "V1": np.random.randn(),
            "V2": np.random.randn(),
            "V3": np.random.randn(),
            "user_id": "sim_user"
        }
        log_transaction(dummy_tx, created_by="sim_user")

        time.sleep(5)

# -----------------------------
# Run App with Simulation
# -----------------------------
if __name__ == "__main__":
    # Start simulation in background
    sim_thread = threading.Thread(target=simulation_worker, daemon=True)
    sim_thread.start()
    
    # Run Flask with threaded mode
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)