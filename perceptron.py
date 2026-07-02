from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate Dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create Perceptron Model
clf = Perceptron(
    max_iter=1000,
    eta0=0.1,
    tol=1e-3,
    shuffle=True,
    random_state=42
)

# Train Model
clf.fit(X_train, y_train)

# Predictions
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("=" * 50)
print("Perceptron Classifier")
print("=" * 50)

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))# from sklearn.linear_model import Perceptron
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# # def step(x):
# #     return 1 if x >= 0 else 0

# # def perceptron(x1, x2, w1, w2, b):
# #     z = x1 * w1 + x2 * w2 + b
# #     return step(z)

# # # Try different weights and bias to match the AND logic
# # print(perceptron(0, 0, 1, 1, -1.5))  # Expected: 0
# # print(perceptron(0, 1, 1, 1, -1.5))  # Expected: 0
# # print(perceptron(1, 0, 1, 1, -1.5))  # Expected: 0
# # print(perceptron(1, 1, 1, 1, -1.5))  # Expected: 1
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# clf = Perceptron(
#     max_iter=1000,    # Maximum number of epochs
#     eta0=0.1,         # Learning rate
#     random_state=42,  # For reproducibility
#     tol=1e-3,         # Stop early if improvement is smaller than this
#     shuffle=True      # Shuffle data each epoch
# )
# clf.fit(X_train, y_train)
# train_accuracy = clf.score(X_train, y_train)
# test_accuracy = clf.score(X_test, y_test)
# print(f"Training Accuracy: {train_accuracy:.4f}")

# accuracy = clf.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.2f}")