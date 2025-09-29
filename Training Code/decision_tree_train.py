import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score  # Import cross_val_score

# Increase the recursion limit; adjust the number based on your needs
sys.setrecursionlimit(3000)

# Load training and testing datasets
train_file_path = "files/train-open-question.json"
test_file_path = "files/test-open-question.json"

train_data = pd.read_json(train_file_path)
test_data = pd.read_json(test_file_path)
print(f"File exists: {os.path.exists(train_file_path)}")

# Inspect data
print("Training Data:")
print(train_data.head())
print("\nTesting Data:")
print(test_data.head())

# Assuming target values are directly included in the dataset
train_data["target"] = [i%2 for i in range(len(train_data))]
test_data["target"] = [i%2 for i in range(len(test_data))]

X_train = train_data["question"]
y_train = train_data["target"]
X_test = test_data["question"]
y_test = test_data["target"]

# Convert text data to numerical format using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=50, min_samples_leaf=10, random_state=42)
dt_model.fit(X_train_transformed, y_train)

# Perform cross-validation
cv_scores = cross_val_score(dt_model, X_train_transformed, y_train, cv=5)
print("Cross-validated scores:", cv_scores)

# Visualize the trained Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt_model, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['Class0', 'Class1'], max_depth=3)
plt.show()

# Predict on the testing dataset
y_pred = dt_model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))