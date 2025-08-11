import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset (update file name if needed)
data = pd.read_csv('app/Data/crop_recommendation.csv')

# Features and label (update column names if different)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']  # Change 'label' if your column is named differently

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model to app/models
os.makedirs('app/models', exist_ok=True)
with open('app/models/RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to app/models/RandomForest.pkl")


