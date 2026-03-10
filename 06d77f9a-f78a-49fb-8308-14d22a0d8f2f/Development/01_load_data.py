import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv", low_memory=False)

# --- First 10 rows ---
print("=" * 60)
print("FIRST 10 ROWS")
print("=" * 60)
print(df.head(10).to_string())

# --- Dataset shape ---
print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# --- Column names ---
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:>3}. {col}")

# --- Data types ---
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

# --- Missing values ---
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# --- Summary statistics ---
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (Numeric Columns)")
print("=" * 60)
print(df.describe())

df