# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style='whitegrid')

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic.head())

# Basic information about the dataset
print(titanic.info())

# Checking for missing values
print(titanic.isnull().sum())

# Data cleaning
# Filling missing Age values with the median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Dropping the Cabin column due to too many missing values
titanic.drop('Cabin', axis=1, inplace=True)

# Dropping rows with missing Embarked values
titanic.dropna(subset=['Embarked'], inplace=True)

# Converting 'Sex' and 'Embarked' to numerical values
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)

# Exploratory Data Analysis (EDA)

# Visualize the distribution of ages
plt.figure(figsize=(10, 6))
sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival rate based on sex
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Survival Rate by Sex')
plt.xticks([0, 1], ['Male', 'Female'])
plt.ylabel('Survival Rate')
plt.show()

# Survival rate based on class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Age vs. Survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=titanic)
plt.title('Age vs Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = titanic.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Conclusions
print("Survival was higher for females and those in higher classes.")
print("Age also shows varying survival rates, indicating potential trends in survival.")