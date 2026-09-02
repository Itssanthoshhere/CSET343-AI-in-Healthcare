"""
Lab Assignment - 2: Data Cleaning and Enrichment on Clinical Dataset
Course: CSET343 - AI in Healthcare | Year: 4th Year, Sem: VII
Dataset: Heart Failure Clinical Records Dataset (UCI Machine Learning Repository)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# -------------------------------------------------------------
# 1. Dataset Acquisition
# -------------------------------------------------------------
# Dataset: Heart Failure Clinical Records Dataset
data_path = "heart_failure_clinical_records_dataset.csv"

# -------------------------------------------------------------
# 2. Load Dataset
# -------------------------------------------------------------
df = pd.read_csv(data_path)
df_before = df.copy() # keep a copy for before/after comparison

print("--- First 5 Rows ---")
print(df.head())

# -------------------------------------------------------------
# 3. Summarize Dataset
# -------------------------------------------------------------
print("\n--- Dataset Summary ---")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values Count:")
print(df.isnull().sum())

# -------------------------------------------------------------
# 4. Visualize Missing Values
# -------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=True, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig("plots/task4_missing_values_heatmap.png")
plt.show()

# -------------------------------------------------------------
# 5. Plot Distributions
# -------------------------------------------------------------
numerical_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                  'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# Histograms for numerical features
df[numerical_cols].hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle("Distributions of Numerical Features (Before Cleaning)")
plt.tight_layout()
plt.savefig("plots/task5_raw_distributions_histograms.png")
plt.show()

# Boxplots for numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['age', 'ejection_fraction', 'serum_sodium']])
plt.title("Boxplots of Selected Numerical Features")
plt.savefig("plots/task5_raw_distributions_boxplots.png")
plt.show()

# -------------------------------------------------------------
# 6. Impute Missing Values
# -------------------------------------------------------------
# Numerical features: Use mean or median
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Categorical features: Use mode or "Unknown"
categorical_cols = ['sex', 'smoking', 'anaemia', 'diabetes', 'high_blood_pressure']
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after imputation check:")
print(df.isnull().sum())

# -------------------------------------------------------------
# 7. Remove Outliers
# -------------------------------------------------------------
# Using IQR method to remove outliers from continuous numerical features
for col in ['ejection_fraction', 'serum_creatinine', 'platelets']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print(f"\nShape after removing outliers: {df.shape}")

# -------------------------------------------------------------
# 8. Correct Inconsistencies
# -------------------------------------------------------------
# Check and fix invalid entries (e.g., negative age)
df = df[df['age'] > 0]
df['age'] = df['age'].astype(int)

# Ensure ejection_fraction is within valid percentage range (0-100)
df = df[(df['ejection_fraction'] >= 0) & (df['ejection_fraction'] <= 100)]
print(f"Shape after checking inconsistencies: {df.shape}")

# -------------------------------------------------------------
# 9. Feature Engineering
# -------------------------------------------------------------
# • Create age_group feature (bins: <40, 40–60, >60)
df['age_group'] = pd.cut(df['age'], bins=[0, 39, 60, 120], labels=['<40', '40–60', '>60'])

# • Create risk_score feature (e.g., normalized product of ejection_fraction, serum_creatinine)
product = df['ejection_fraction'] * df['serum_creatinine']
df['risk_score'] = (product - product.min()) / (product.max() - product.min())

print("\nEngineered Features (age_group & risk_score):")
print(df[['age', 'age_group', 'ejection_fraction', 'serum_creatinine', 'risk_score']].head())

# -------------------------------------------------------------
# 10. Encode Categorical Variables
# -------------------------------------------------------------
# Apply label encoding to age_group
le = LabelEncoder()
df['age_group_encoded'] = le.fit_transform(df['age_group'])

# Apply one-hot encoding to categorical features (e.g., sex, smoking, age_group)
df = pd.get_dummies(df, columns=['sex', 'smoking', 'age_group'], drop_first=False)
print("\nColumns after encoding:")
print(df.columns.tolist())

# -------------------------------------------------------------
# 11. Normalize Features
# -------------------------------------------------------------
scaler = MinMaxScaler()
scale_cols = ['age', 'platelets', 'creatinine_phosphokinase', 'serum_sodium', 'time', 'ejection_fraction', 'serum_creatinine']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("\nSample normalized features:")
print(df[scale_cols].head())

# -------------------------------------------------------------
# 12. Compute Summary Statistics (Before vs. After Cleaning)
# -------------------------------------------------------------
print("\n--- Summary Statistics (Before Cleaning) ---")
stats_before = df_before[numerical_cols].agg(['mean', 'median', 'std']).T
print(stats_before.round(2))

print("\n--- Summary Statistics (After Cleaning) ---")
# Re-compute unscaled numerical columns from cleaned subset for direct comparison
stats_after = df[scale_cols].agg(['mean', 'median', 'std']).T
print(stats_after.round(2))

# -------------------------------------------------------------
# 13. Validate Cleaning
# -------------------------------------------------------------
# Plot histograms for cleaned numerical features
df[scale_cols].hist(figsize=(10, 8), bins=15, edgecolor='black', color='teal')
plt.suptitle("Distributions of Cleaned & Normalized Features")
plt.tight_layout()
plt.savefig("plots/task13_validation_cleaning_distributions.png")
plt.show()

# -------------------------------------------------------------
# 14. Check Missing Values
# -------------------------------------------------------------
print("\n--- Final Check for Missing Values ---")
print(df.isnull().sum())
print(f"\nTotal missing values remaining: {df.isnull().sum().sum()}")
print("Result: Confirmed no missing values remain in the dataset.")

# Save cleaned dataset
df.to_csv("heart_failure_cleaned.csv", index=False)
print("Saved cleaned dataset to heart_failure_cleaned.csv")
