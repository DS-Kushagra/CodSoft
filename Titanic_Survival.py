import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

# Load the dataset
data = pd.read_csv("C:\\Data Science\\CodSoft Internship\\Titanic-Dataset.csv")
print(data.describe())

# Dropping the following columns
data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Heatmap for missing data
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Fixing Categorical values
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
print(data.isnull().sum())

# Countplot for Survived
print(data['Survived'].value_counts())
sns.set_style("whitegrid")
sns.countplot(x="Survived", data=data)
plt.title('Survival Count')
plt.show()

# Countplot for Survived by Sex
print(data['Sex'].value_counts())
sns.countplot(x="Survived", hue="Sex", data=data)
plt.title('Survival by Gender')
plt.show()

# Countplot for Survived by Pclass
sns.countplot(x="Survived", hue="Pclass", data=data, palette='rainbow')
plt.title('Survival by Passenger Class')
plt.show()

# Survived by Age
sns.violinplot(x="Survived", y="Age", data=data, palette="muted")
plt.title('Survival Distribution by Age')
plt.show()

# Encoding categorical features
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
print(data.head())

# Splitting the data
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
log = LogisticRegression(random_state=0)
log.fit(X_train, y_train)

# Making predictions
pred = log.predict(X_test)
print(pred)
print(y_test.values)

# Suppress warnings
warnings.filterwarnings("ignore")

# Example prediction
res = log.predict([[2, 0, 25, 50, 0]])  # Example input: Pclass=2, Sex=0, Age=25, Fare=50, Embarked=0
print("Survived" if res == 1 else "Not Survived")