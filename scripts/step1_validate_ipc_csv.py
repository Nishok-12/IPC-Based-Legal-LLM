import pandas as pd

input_path = "../data/ipc_clean.csv"
output_path = "../data/ipc_step1_validated.csv"

# Read CSV with correct encoding
df = pd.read_csv(input_path, encoding="latin1")

print("CSV loaded successfully")
print("Columns:", df.columns.tolist())
print("Rows before cleaning:", len(df))

df = df.dropna(how="all")
df = df.drop_duplicates()

print("\nMissing values per column:")
print(df.isnull().sum())

df = df.fillna("")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.replace("\n", " ").str.strip()

print("Rows after cleaning:", len(df))

df.to_csv(output_path, index=False, encoding="utf-8")

print("\nSTEP 1 completed successfully")
print("Saved file:", output_path)
