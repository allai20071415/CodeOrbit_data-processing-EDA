import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ecommerce_sales.csv")
print("Initial shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["Region"] = df["Region"].fillna(df["Region"].mode()[0])

print("\nClean shape:", df.shape)
print("\nSummary statistics:\n", df.describe())

df.groupby("Category")["Sales"].sum().sort_values(ascending=False).plot(kind="bar")
plt.title("Total Sales by Category"); plt.xlabel("Category"); plt.ylabel("Sales")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()
