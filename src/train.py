import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


# Set up command-line arguments
parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset")
parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing (default: 0.2)")
parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")
args = parser.parse_args()

test_size = args.test_size
random_state = args.random_state


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")

# Initialize and train the Decision Tree
model = DecisionTreeClassifier(random_state=random_state)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.2f}")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot and save the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
print("Confusion matrix saved to outputs/confusion_matrix.png")

# Save the trained model
joblib.dump(model, "outputs/model.joblib")
print("Trained model saved to outputs/model.joblib")


