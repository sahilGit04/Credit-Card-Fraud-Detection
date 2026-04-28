import numpy as np

def simulate_transaction():
    amount = np.random.uniform(1, 5000)

    # Simulate fraud pattern
    if np.random.rand() < 0.02:
        return {
            "amount": amount * 5,
            "is_fraud": 1
        }
    else:
        return {
            "amount": amount,
            "is_fraud": 0
        }

def generate_transactions(n=10):
    return [simulate_transaction() for _ in range(n)]