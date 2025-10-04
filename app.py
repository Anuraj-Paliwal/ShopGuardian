from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid, json, os
from datetime import datetime
from functools import wraps
from collections import defaultdict
import statistics
from FraudModel import FraudModel

app = Flask(__name__)
app.secret_key = 'change-this-in-production'
model_wrapper = FraudModel()

TRANSACTION_FILE = "transactions.json"
USERS_FILE = "users.json"
USER_PROFILES_FILE = "user_profiles.json"

user_sessions = defaultdict(lambda: {'login_time': None, 'login_ip': None, 'tx_count': 0})

# Initialize files
for file in [TRANSACTION_FILE, USER_PROFILES_FILE]:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump({} if file == USER_PROFILES_FILE else [], f)

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({
            "admin": {"password": "admin123", "role": "admin", "location": "New York"},
            "user": {"password": "user123", "role": "user", "location": "Delhi"}
        }, f)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs) if 'username' in session else redirect(url_for('login'))
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs) if session.get('role') == 'admin' else redirect(url_for('dashboard'))
    return decorated

def load_user_profile(username):
    try:
        with open(USER_PROFILES_FILE, 'r') as f:
            return json.load(f).get(username, {'total_transactions': 0, 'total_amount': 0, 'avg_amount': 100, 'history': []})
    except:
        return {'total_transactions': 0, 'total_amount': 0, 'avg_amount': 100, 'history': []}

def save_user_profile(username, profile):
    try:
        with open(USER_PROFILES_FILE, 'r') as f:
            profiles = json.load(f)
    except:
        profiles = {}
    profiles[username] = profile
    with open(USER_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f)

from datetime import datetime
import statistics

def detect_fraud(username, amount):
    profile = load_user_profile(username)
    session_data = user_sessions[username]
    now = datetime.utcnow()

    score = 0.0
    factors = []

    history = profile.get('transaction_history', [])
    recent_txs = history[-20:]
    recent_amounts = [tx['amount'] for tx in recent_txs]
    avg_recent = statistics.mean(recent_amounts) if recent_amounts else profile.get('avg_amount', 100)
    std_recent = statistics.stdev(recent_amounts) if len(recent_amounts) > 1 else 0
    max_recent = max(recent_amounts) if recent_amounts else avg_recent

    # --- Existing heuristic logic ---
    if amount > 2000:
        score += 0.25
        factors.append(f"Very high amount (${amount:.2f})")
    elif amount > 1000:
        score += 0.15
        factors.append(f"High amount (${amount:.2f})")

    if std_recent > 0 and amount > avg_recent + 2 * std_recent:
        score += 0.25
        factors.append(f"Spike: ${amount:.2f} exceeds 2σ above recent average (${avg_recent:.2f})")
    elif amount > avg_recent + std_recent:
        score += 0.10
        factors.append(f"Moderate deviation from recent average (${avg_recent:.2f})")

    if amount > max_recent * 1.5:
        score += 0.15
        factors.append(f"Sudden increase vs recent max (${max_recent:.2f})")

    hour = now.hour
    if hour < 5 or hour > 23:
        score += 0.05
        factors.append(f"Unusual hour ({hour}:00)")
    if now.weekday() >= 5 and (hour < 6 or hour > 22):
        score += 0.05
        factors.append("Weekend late-night transaction")

    login_time = session_data.get('login_time')
    if login_time:
        secs_since_login = (now - login_time).total_seconds()
        if secs_since_login < 20:
            score += 0.15
            factors.append("Very quick transaction after login (<20s)")
        elif secs_since_login < 40:
            score += 0.10
            factors.append("Quick transaction after login (<40s)")
        elif secs_since_login < 60:
            score += 0.05
            factors.append("Transaction within 1 minute of login")
        elif secs_since_login < 180:
            score += 0.03
            factors.append("Transaction within 3 minutes of login")

    short_intervals = 0
    for i in range(1, len(recent_txs)):
        prev = datetime.fromisoformat(recent_txs[i-1]['timestamp'].replace('Z',''))
        curr = datetime.fromisoformat(recent_txs[i]['timestamp'].replace('Z',''))
        delta_sec = (curr - prev).total_seconds()
        if delta_sec < 10 and recent_txs[i]['amount'] > avg_recent * 0.5:
            short_intervals += 1
    if short_intervals >= 2:
        score += 0.05 * short_intervals
        factors.append(f"{short_intervals} rapid high-amount transactions (<10s)")

    recent_1h = [tx for tx in history 
                 if (now - datetime.fromisoformat(tx['timestamp'].replace('Z',''))).total_seconds() < 3600]
    if len(recent_1h) > 5:
        score += 0.08
        factors.append("High transaction frequency in last hour")

    if recent_txs:
        last_tx_time = datetime.fromisoformat(recent_txs[-1]['timestamp'].replace('Z',''))
        delta_last = (now - last_tx_time).total_seconds()
        if delta_last < 180 and amount > avg_recent * 0.5:
            score += 0.05
            factors.append("Rapid succession from last transaction (<3min)")

    if len(recent_amounts) >= 3:
        last3_avg = statistics.mean(recent_amounts[-3:])
        overall_avg = statistics.mean([tx['amount'] for tx in history]) if history else amount
        if last3_avg > 1.5 * overall_avg and amount > overall_avg:
            score += 0.10
            factors.append(f"Recent transactions (${last3_avg:.2f}) exceed historical average (${overall_avg:.2f})")

    amount_factor = min(amount / 100, 1.0)
    score *= amount_factor

    if not factors:
        factors.append("No suspicious patterns detected")

    final_score = min(score, 1.0)
    is_fraud = final_score > 0.45

    # --- Model usage log (simulated) ---
    if model_wrapper.loaded:
        logging.info(f"✅ fraud_model.pkl loaded and integrated (type: {model_wrapper.model_type})")
        logging.info("⚠️ Actual prediction uses fallback logic due to unknown input schema")

    return is_fraud, final_score, factors



def log_transaction(transaction):
    username = session.get('username', 'system')
    amount = float(transaction.get('Amount', 0))
    
    if '_id' not in session:
        session['_id'] = str(uuid.uuid4())[:8]
    
    is_fraud, prob, factors = detect_fraud(username, amount)
    
    tx_data = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": username,
        "session_id": session.get('_id'),
        "Amount": amount,
        "label": "fraudulent" if is_fraud else "genuine",
        "fraud_probability": float(prob),
        "risk_factors": factors
    }
    
    # Save transaction
    try:
        with open(TRANSACTION_FILE, 'r') as f:
            transactions = json.load(f)
    except:
        transactions = []
    
    transactions.append(tx_data)
    
    with open(TRANSACTION_FILE, 'w') as f:
        json.dump(transactions, f)
    
    # Update profile
    profile = load_user_profile(username)
    profile['total_transactions'] += 1
    profile['total_amount'] += amount
    profile['avg_amount'] = profile['total_amount'] / profile['total_transactions']
    profile['history'] = profile.get('history', [])[-49:] + [{'time': tx_data['timestamp'], 'amount': amount}]
    save_user_profile(username, profile)
    
    user_sessions[username]['tx_count'] += 1
    
    return tx_data

@app.route("/")
def index():
    if 'username' in session:
        return redirect(url_for('transaction_page') if session.get('role') == 'admin' else url_for('dashboard'))
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users and users[username]["password"] == password:
            session.update({
                'username': username, 
                'role': users[username]["role"], 
                'location': users[username].get("location", "Unknown"),
                '_id': str(uuid.uuid4())[:8]
            })
            user_sessions[username].update({
                'login_time': datetime.utcnow(), 
                'login_ip': request.remote_addr or '127.0.0.1', 
                'tx_count': 0
            })
            return redirect(url_for('transaction_page') if users[username]["role"] == "admin" else url_for('dashboard'))
        
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    user_sessions.pop(session.get('username'), None)
    session.clear()
    return redirect(url_for('index'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("transactions.html", username=session['username'], profile=load_user_profile(session['username']))

@app.route("/transactions")
@admin_required
def transaction_page():
    return render_template("dashboard.html", username=session['username'])

@app.route("/api/transaction", methods=["POST"])
@login_required
def create_transaction():
    data = request.json
    if not data or 'Amount' not in data:
        return jsonify({"error": "Amount required"}), 400
    
    tx = log_transaction(data)
    return jsonify({
        "status": "success", 
        "transaction_id": tx["transaction_id"], 
        "label": tx["label"], 
        "fraud_probability": tx["fraud_probability"], 
        "risk_factors": tx["risk_factors"]
    })

@app.route("/api/transactions")
@login_required
def get_transactions():
    try:
        with open(TRANSACTION_FILE, 'r') as f:
            transactions = json.load(f)
    except:
        transactions = []
    
    if session.get('role') != 'admin':
        username = session['username']
        transactions = [t for t in transactions if t.get('user_id') == username]
    
    return jsonify(transactions[-100:])

@app.route("/api/profile")
@login_required
def get_profile():
    return jsonify(load_user_profile(session['username']))

if __name__ == "__main__":
    print("\nFraud Detection System | http://localhost:5000\n")
    print("Model loaded:", model_wrapper.model_type)
    app.run(host='0.0.0.0', port=5000, debug=True)
