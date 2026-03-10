import pandas as pd

cleaned_df = df.copy()

# Convert timestamp
cleaned_df["timestamp"] = pd.to_datetime(
    cleaned_df["timestamp"], utc=True, errors="coerce"
)

# Remove rows with invalid timestamps
cleaned_df = cleaned_df.dropna(subset=["timestamp"])

# Sort by user timeline
cleaned_df = cleaned_df.sort_values(
    ["distinct_id", "timestamp"]
).reset_index(drop=True)

# Ensure credits numeric
cleaned_df["prop_credits_used"] = pd.to_numeric(
    cleaned_df["prop_credits_used"], errors="coerce"
).fillna(0)

# Missing values report
_missing = cleaned_df.isnull().sum()
_missing_cols = _missing[_missing > 0]

print("=" * 60)
print("MISSING VALUES (columns with nulls only)")
print("=" * 60)

if _missing_cols.empty:
    print("No missing values found!")
else:
    _missing_pct = (_missing_cols / len(cleaned_df) * 100).round(2)
    _missing_report = pd.DataFrame({
        "Missing Count": _missing_cols,
        "Missing %": _missing_pct
    })
    print(_missing_report)

# Remove duplicates
_before = len(cleaned_df)
cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
_removed = _before - len(cleaned_df)

print("\n" + "=" * 60)
print("DEDUPLICATION SUMMARY")
print("=" * 60)
print(f"Rows before : {_before:,}")
print(f"Duplicates  : {_removed:,}")
print(f"Rows after  : {len(cleaned_df):,}")

cleaned_df