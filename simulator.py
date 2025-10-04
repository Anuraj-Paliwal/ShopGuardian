"""
Standalone Transaction Simulator
Generates transactions directly to JSON file (no Flask needed)
Perfect for batch data generation and testing
"""

import json
import time
import random
import numpy as np
import uuid
from datetime import datetime, timedelta

class StandaloneSimulator:
    def __init__(self, output_file="transactions.json"):
        self.output_file = output_file
        self.transaction_count = 0
        self.start_time = None
        self.fraud_count = 0
        self.genuine_count = 0
        
        # Simulation parameters
        self.users = [f"user_{i}" for i in range(1, 51)]
        self.locations = ["India", "USA", "UK", "Germany", "Canada", "Australia", "Japan"]
        self.payment_methods = ["credit_card", "debit_card", "paypal", "upi", "apple_pay"]
        
    def generate_realistic_transaction(self):
        """Generate a transaction with realistic patterns"""
        
        # Decide if fraud (10% base rate, varies by time)
        current_hour = datetime.now().hour
        
        # Fraud more likely late at night (2-5 AM)
        if 2 <= current_hour <= 5:
            fraud_rate = 0.25
        # Normal hours
        elif 9 <= current_hour <= 20:
            fraud_rate = 0.08
        # Evening
        else:
            fraud_rate = 0.15
            
        is_fraud = random.random() < fraud_rate
        
        # Generate realistic Amount
        if is_fraud:
            amount_pattern = random.choice(['small', 'medium', 'large'])
            if amount_pattern == 'small':
                amount = random.uniform(1, 50)
            elif amount_pattern == 'medium':
                amount = random.uniform(200, 800)
            else:
                amount = random.uniform(1500, 5000)
        else:
            amount = abs(np.random.normal(120, 180))
            amount = min(amount, 3000)
        
        # Calculate Time
        if self.start_time is None:
            elapsed_time = 0
        else:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        # Generate V1-V28
        v_features = {}
        for i in range(1, 29):
            if is_fraud:
                v_features[f"V{i}"] = round(np.random.normal(0, 3.5), 6)
            else:
                v_features[f"V{i}"] = round(np.random.normal(0, 1.8), 6)
        
        # Location patterns
        if is_fraud:
            location = random.choice(["Russia", "Nigeria", "China", "Unknown", "Proxy"])
        else:
            location = random.choice(self.locations)
        
        # Don't set label here - let ML model decide!
        # Just track what we INTENDED (for statistics)
        intended_fraud = is_fraud
        
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": random.choice(self.users),
            "Amount": round(amount, 2),
            "Time": round(elapsed_time, 2),
            "location": location,
            "payment_method": random.choice(self.payment_methods),
            # Don't include label/probability - these should come from ML model
            # But standalone doesn't have ML, so we estimate:
            "label": "fraudulent" if is_fraud else "genuine",
            "fraud_probability": random.uniform(0.7, 0.99) if is_fraud else random.uniform(0.01, 0.3),
            "created_by": "simulator",
            **v_features
        }
        
        # Note: If you want TRUE ML predictions, use the admin panel simulator instead
        # This standalone version is for quick data generation
        
        if is_fraud:
            self.fraud_count += 1
        else:
            self.genuine_count += 1
        
        return transaction, is_fraud
    
    def save_transaction(self, transaction):
        """Save transaction to JSON file"""
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(transaction) + "\n")
            return True
        except Exception as e:
            print(f"❌ Error saving transaction: {e}")
            return False
    
    def run_simulation(self, num_transactions=100, delay=0.5):
        """Run simulation"""
        self.start_time = datetime.now()
        self.transaction_count = 0
        self.fraud_count = 0
        self.genuine_count = 0
        
        print(f"\n{'='*70}")
        print(f"🚀 STANDALONE TRANSACTION SIMULATOR")
        print(f"{'='*70}")
        print(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Generating {num_transactions} transactions")
        print(f"💾 Output file: {self.output_file}")
        print(f"⏱️  Delay: {delay}s between transactions")
        print(f"{'='*70}\n")
        
        for i in range(1, num_transactions + 1):
            transaction, is_fraud = self.generate_realistic_transaction()
            
            if self.save_transaction(transaction):
                self.transaction_count += 1
                
                # Log result
                status_icon = "🚨" if is_fraud else "✅"
                status_text = "FRAUDULENT" if is_fraud else "GENUINE"
                
                print(f"{status_icon} [{i}/{num_transactions}] "
                      f"Amount=${transaction['Amount']:.2f} | "
                      f"{status_text} ({transaction['fraud_probability']*100:.1f}%) | "
                      f"User: {transaction['user_id']} | "
                      f"Location: {transaction['location']}")
            
            time.sleep(delay)
        
        # Summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        fraud_rate = (self.fraud_count / self.transaction_count * 100) if self.transaction_count > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🏁 SIMULATION COMPLETED")
        print(f"{'='*70}")
        print(f"📊 Total Transactions: {self.transaction_count}")
        print(f"🚨 Fraudulent: {self.fraud_count} ({fraud_rate:.1f}%)")
        print(f"✅ Genuine: {self.genuine_count} ({100-fraud_rate:.1f}%)")
        print(f"⏱️  Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"💾 Saved to: {self.output_file}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_transactions = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    output_file = sys.argv[3] if len(sys.argv) > 3 else "transactions.json"
    
    print(f"📝 Configuration:")
    print(f"   Transactions: {num_transactions}")
    print(f"   Delay: {delay}s")
    print(f"   Output: {output_file}")
    print()
    
    simulator = StandaloneSimulator(output_file=output_file)
    
    try:
        simulator.run_simulation(num_transactions=num_transactions, delay=delay)
        print("✅ Simulation completed successfully!")
        print(f"💡 You can now run 'python app.py' to view transactions in the dashboard")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Simulation interrupted by user")
        print(f"📊 Generated {simulator.transaction_count} transactions before stopping")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")