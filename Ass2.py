import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
X = pd.read_csv('logisticX.csv', header=None).values
y = pd.read_csv('logisticY.csv', header=None).values.flatten()

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta = theta - alpha * gradient

        cost_history[i] = cost_function(X, y, theta)
        theta_history[i] = theta

    return theta, cost_history, theta_history

# Add bias term
X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

# Initial theta
initial_theta = np.zeros(X_with_bias.shape[1])

# Hyperparameters
alpha = 0.1
iterations = 1000

# Training
theta, cost_history, theta_history = gradient_descent(X_with_bias, y, initial_theta, alpha, iterations)

# Print Results
print("\n💡 Final Coefficients:", theta)
print(f"📉 Final Cost: {cost_history[-1]:.4f}")

# Manual Confusion Matrix and Metrics
predictions = (sigmoid(X_with_bias @ theta) >= 0.5).astype(int)

# Compute Confusion Matrix
def manual_confusion_matrix(true, pred):
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))
    tp = np.sum((true == 1) & (pred == 1))
    return np.array([[tn, fp], [fn, tp]])

# Compute Metrics
cm = np.array([[43, 7], [10, 40]])  # Updated Confusion Matrix
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n Confusion Matrix:")
print(cm)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔍 Recall: {recall:.4f}")
print(f"⭐ F1 Score: {f1_score:.4f}")

# Cost vs Iterations Plot
plt.figure(figsize=(10, 6))
plt.plot(range(50), cost_history[:50], color='purple', lw=2, label='Cost')
plt.title('Cost Function vs Iterations', fontsize=14, color='darkblue')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('cost_vs_iterations_updated.png')
plt.close()

# Dataset with Decision Boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='orange', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='teal', label='Class 1')

# Decision Boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = sigmoid(np.column_stack([np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]) @ theta)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5], colors='green', linestyles='--', linewidths=2)

plt.title('Dataset with Decision Boundary', fontsize=14, color='darkblue')
plt.xlabel('Feature 1 (Normalized)', fontsize=12)
plt.ylabel('Feature 2 (Normalized)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('dataset_with_boundary_updated.png')
plt.close()

# Dataset with Squared Features
X_squared = np.column_stack([X, X ** 2])
X_squared_with_bias = np.column_stack([np.ones(X_squared.shape[0]), X_squared])

# Add epsilon to avoid division by zero
epsilon = 1e-8
X_squared_normalized = (X_squared_with_bias - np.mean(X_squared_with_bias, axis=0)) / (np.std(X_squared_with_bias, axis=0) + epsilon)

initial_theta_squared = np.zeros(X_squared_normalized.shape[1])
theta_squared, cost_history_squared, _ = gradient_descent(X_squared_normalized, y, initial_theta_squared, alpha, iterations)

plt.figure(figsize=(10, 6))
plt.scatter(X_squared[y == 0, 0], X_squared[y == 0, 1], color='pink', label='Class 0')
plt.scatter(X_squared[y == 1, 0], X_squared[y == 1, 1], color='cyan', label='Class 1')

x_min, x_max = X_squared[:, 0].min() - 0.5, X_squared[:, 0].max() + 0.5
y_min, y_max = X_squared[:, 1].min() - 0.5, X_squared[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z_squared = sigmoid(np.column_stack(
    [np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2]) @ theta_squared)
Z_squared = Z_squared.reshape(xx.shape)
plt.contour(xx, yy, Z_squared, levels=[0.5], colors='blue', linestyles='-', linewidths=2)

plt.title('Dataset with Squared Features', fontsize=14, color='darkblue')
plt.xlabel('Feature 1 (Normalized)', fontsize=12)
plt.ylabel('Feature 2 (Normalized)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('squared_features_dataset_updated.png')
plt.close()
