import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set(style="whitegrid")
df = pd.read_csv("C:/Users/STEFANA DCRUZ/Downloads/Titanic-Dataset.csv")
print(df.head(), "\n")

print(" Dataset Information:")
df.info()
print("\n")

print(" Missing Values:")
print(df.isnull().sum(), "\n")

print(" Summary Statistics:")
print(df.describe(include='all'), "\n")

print(" Unique Values (for categorical features):")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
'''
print("\nCreating Histograms...")
df[numeric_cols].hist(bins=30, figsize=(15, 10), color='#1f77b4')
plt.suptitle("Histogram of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

print("\n Creating Boxplots...")
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, len(numeric_cols) // 2 + 1, i + 1)
    sns.boxplot(x=df[col], color='salmon')
    plt.title(col)
plt.suptitle("Boxplots of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

print("\n Correlation Matrix...")
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

print("\n Creating Pairplot (can be slow with many features)...")
sns.pairplot(df[numeric_cols])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

print("\n Interactive Scatter Matrix with Plotly...")
fig = px.scatter_matrix(df, dimensions=numeric_cols, color=df[categorical_cols[0]] if len(categorical_cols) > 0 else None)
fig.update_layout(title="Interactive Scatter Matrix", width=1000, height=800)
fig.show()

print("\n Exploring Feature Distributions & Relationships...")
g = sns.PairGrid(df[numeric_cols])
g.map_diag(sns.histplot, kde=True)
g.map_offdiag(sns.scatterplot)
plt.show()
'''
print("\nOutlier Detection Summary:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} potential outliers")