import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv(r"C:\Data Science\CodSoft Internship\Task 3\IRIS.csv")

# Display the first few rows and data types
print("Data types of columns:")
print(df.dtypes)
print("First few rows of the dataset:")
print(df.head())

# Convert all feature columns to numeric
for column in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Convert categorical target variable to numeric
y = pd.factorize(y)[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the k-NN classifier
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

