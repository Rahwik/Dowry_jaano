import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv('data/dowry_data.csv')

# Preprocess the data
label_encoders = {}
categorical_cols = ['income', 'education', 'location_type', 'region', 'caste', 'religion']

# Convert categorical columns into numeric codes
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset into features (X) and target (y)
X = df.drop('dowry_amount', axis=1)
y = df['dowry_amount']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")

# Save the model
with open('model/dowry_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the label encoders
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and label encoders saved to 'model/' directory.")
