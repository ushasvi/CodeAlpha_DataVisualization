import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 1. Survival Distribution
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.savefig("survival_distribution.png")
plt.show()

# 2. Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_gender.png")
plt.show()

# 3. Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.savefig("survival_class.png")
plt.show()

# 4. Age Distribution
sns.histplot(df['Age'].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# 5. Fare Distribution
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.savefig("fare_distribution.png")
plt.show()

# 6. Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()
