import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess the dataset
data = pd.read_csv(r"C:\Data Science\CodSoft Internship\Task 2\Movies-Rating.csv", encoding='ISO-8859-1')

# Fill missing values
data.fillna({'Genre': 'Unknown', 'Director': 'Unknown', 'Actor 1': 'Unknown', 
             'Actor 2': 'Unknown', 'Actor 3': 'Unknown',
             'Votes': data['Votes'].median(),
             'Duration': data['Duration'].median(),
             'Year': data['Year'].median()}, inplace=True)

# Drop rows with missing ratings
data.dropna(subset=['Rating'], inplace=True)

# Drop the 'Name' column
data.drop(['Name'], axis=1, inplace=True)

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_categorical = encoder.fit_transform(data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']])

# Combine features and target variable
X = np.hstack((encoded_categorical.toarray(), data[['Year', 'Duration', 'Votes']].values))
y = data['Rating']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train the model and make predictions
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Predict the rating for a new movie entry
example_data = pd.DataFrame({
    'Year': [2005],
    'Duration': [142],
    'Genre': ['Drama Romance War'],
    'Votes': [1086],
    'Director': ['Shoojit Sircar'],
    'Actor 1': ['Jimmy Sheirgill'],
    'Actor 2': ['issha Lamba'],
    'Actor 3': ['Yashpal Sharma']
})

# Encode and scale the example data
encoded_example = encoder.transform(example_data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']])
example_data_encoded = np.hstack((encoded_example.toarray(), example_data[['Year', 'Duration', 'Votes']].values))
example_data_scaled = scaler.transform(example_data_encoded)

# Predict and print the rating
predicted_rating = model.predict(example_data_scaled)
print(f"Predicted Rating: {predicted_rating[0]:.2f}")
