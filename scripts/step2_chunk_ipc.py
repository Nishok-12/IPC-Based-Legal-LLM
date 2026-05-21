import pandas as pd
import uuid

# Input and output paths
input_path = "../data/ipc_step1_validated.csv"
output_path = "../data/ipc_chunks.csv"

# Load validated CSV
df = pd.read_csv(input_path)

chunks = []

for _, row in df.iterrows():
    section_number = row["Section Number"]
    section_text = row["Section Text"]

    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "section_number": section_number,
        "chunk_text": section_text
    }

    chunks.append(chunk)

# Convert to DataFrame
chunks_df = pd.DataFrame(chunks)

# Save chunks
chunks_df.to_csv(output_path, index=False, encoding="utf-8")

print("STEP 2 completed successfully")
print("Total chunks created:", len(chunks_df))
print("Saved file:", output_path)
