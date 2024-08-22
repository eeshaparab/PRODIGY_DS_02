import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
file_path = r"C:\Users\eesha\OneDrive\Desktop\prodigy\Task2\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Initial Data:")
print(df.head())

# Step 1: Data Cleaning

# 1.1 Checking for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# 1.2 Handling missing values

# Dropping the 'Cabin' column due to excessive missing values
df = df.drop(columns=['Cabin'])

# Filling missing 'Age' values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Filling missing 'Embarked' values with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Verifying that there are no more missing values
missing_values_after = df.isnull().sum()
print("\nMissing Values after Cleaning:")
print(missing_values_after)

# Step 2: Exploratory Data Analysis (EDA)

# 2.1 Distribution of Categorical Variables
plt.figure(figsize=(14, 7))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Class Distribution and Survival')
plt.show()

plt.figure(figsize=(14, 7))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Gender Distribution and Survival')
plt.show()

plt.figure(figsize=(14, 7))
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Embarkation Port Distribution and Survival')
plt.show()

# 2.2 Age Distribution
plt.figure(figsize=(14, 7))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# 2.3 Survival by Age
plt.figure(figsize=(14, 7))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Survival by Age')
plt.show()

# 2.4 Correlation Heatmap
plt.figure(figsize=(14, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 2.5 Exploring Relationships
# Relationship between Fare and Survival
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='Survived', y='Fare')
plt.title('Fare vs Survival')
plt.show()

# Relationship between Age, Fare, and Survival
plt.figure(figsize=(14, 7))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette='coolwarm')
plt.title('Age vs Fare by Survival')
plt.show()
