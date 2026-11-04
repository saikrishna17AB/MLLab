import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: SOFTMAX
# ============================================================

def softmax(logits):

    exp_values = np.exp(logits)

    probabilities = exp_values / np.sum(exp_values)

    return probabilities


# ============================================================
# STEP 2: CROSS-ENTROPY
# ============================================================

def cross_entropy(p, q):

    epsilon = 1e-15

    # Prevent log(0)
    q = np.clip(q, epsilon, 1.0)

    loss = -np.sum(p * np.log2(q))

    return loss


# ============================================================
# STEP 2: KL DIVERGENCE
# ============================================================

def kl_divergence(p, q):

    epsilon = 1e-15

    # Prevent log(0)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    loss = np.sum(p * np.log2(p / q))

    return loss


# ============================================================
# STEP 3: TRUE LABEL
# ============================================================

# Class 0 is the correct class
p = np.array([1, 0, 0])


# ============================================================
# STEP 3: SIMULATE PREDICTIONS
# ============================================================

simulated_logits = [
    np.array([w, 1.0, 0.5])
    for w in np.linspace(-2, 5, 100)
]


# ============================================================
# STEP 3: CALCULATE LOSS VALUES
# ============================================================

cross_entropy_losses = []
kl_losses = []

for logits in simulated_logits:

    # Convert logits into probabilities
    q = softmax(logits)

    # Calculate Cross-Entropy
    ce = cross_entropy(p, q)

    # Calculate KL Divergence
    kl = kl_divergence(p, q)

    # Store the results
    cross_entropy_losses.append(ce)
    kl_losses.append(kl)


# ============================================================
# DISPLAY SOME VALUES
# ============================================================

print("First prediction:")
print("Logits:", simulated_logits[0])
print("Probabilities:", softmax(simulated_logits[0]))
print("Cross-Entropy:", cross_entropy_losses[0])
print("KL Divergence:", kl_losses[0])

print("\nLast prediction:")
print("Logits:", simulated_logits[-1])
print("Probabilities:", softmax(simulated_logits[-1]))
print("Cross-Entropy:", cross_entropy_losses[-1])
print("KL Divergence:", kl_losses[-1])


# ============================================================
# STEP 4: PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cross_entropy_losses,
    label="Cross-Entropy"
)

plt.plot(
    kl_losses,
    label="KL Divergence"
)

plt.xlabel("Prediction Step")
plt.ylabel("Loss")

plt.title("Cross-Entropy vs KL Divergence")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()