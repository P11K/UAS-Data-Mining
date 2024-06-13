import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv('Predict Students Dropout and Academic Success.csv', delimiter=';')

# Display first few rows to understand the data structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handling missing values - fill with mean for numerical columns
df.fillna(df.mean(), inplace=True)

# Encoding categorical variables if any
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Assuming 'Target' is the column to predict and contains 1 for dropout and 0 for no dropout
X = df.drop('Target', axis=1)
y = df['Target']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler and label encoders as well for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
