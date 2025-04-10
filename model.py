import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('Traffic.csv')

# Define the vehicle columns (fix starts here)
vehicle_cols = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']

# Features and labels
X = df[vehicle_cols]
y = df['Traffic Situation']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


import joblib

# After training the model
joblib.dump(model, 'traffic_model.pkl')
