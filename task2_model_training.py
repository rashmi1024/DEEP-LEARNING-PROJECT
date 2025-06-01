

"""This script loads the processed training data (created in Task 1),
builds a Random Forest classifier, evaluates its predictions, and
exports both the results and the trained model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step A: Load input datasets
try:
    X_train = pd.read_csv('train_features_v2.csv')
    X_test = pd.read_csv('test_features_v2.csv')
    y_train = pd.read_csv('train_labels_v2.csv')['label']
    y_test = pd.read_csv('test_labels_v2.csv')['label']
    print("✅ All CSV files loaded successfully.")
except Exception as e:
    print("❌ File loading error:", e)

# Step B: Create Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=99)
print("🧠 Random Forest model created.")

# Step C: Fit model with training data
model_rf.fit(X_train, y_train)
print("✅ Model training completed.")

# Step D: Predict churn on test data
predictions = model_rf.predict(X_test)
print("📊 Predictions made on test set.")


# Step E: Check how well the model performed
accuracy = accuracy_score(y_test, predictions)
print(f"\n🎯 Accuracy: {accuracy:.2f}")
print("\n📄 Classification Report:\n")
print(classification_report(y_test, predictions))

# Step F: Save prediction comparison
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})
results.to_csv('churn_predictions_v2.csv', index=False)
print("📁 Predictions saved as 'churn_predictions_v2.csv'")


#  Save the Model
# Step G: Save trained model to file
joblib.dump(model_rf, 'churn_model_v2.pkl')
print("💾 Trained model saved as 'churn_model_v2.pkl'")