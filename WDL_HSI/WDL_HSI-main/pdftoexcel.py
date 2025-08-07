import os
import re
import pandas as pd

from pathlib import Path

# === CONFIGURATION ===
root_dir = Path(r"C:\Users\joshu\Documents\Coding Projects\WDL\WDL_HSI\WDL_HSI-main\WDLProject-main\Tests\UOTTestsSalinas")  # ← Change this
excel_file = Path("experiment_data.xlsx")    # ← Output file (will be created if doesn't exist)

# === REGEX PATTERNS ===
folder_pattern = re.compile(r"UOT - big_fixed_sample_k=(\d+)_mu=(\d+)_reg=([\d.]+)")
file_pattern = re.compile(r"learned_loose_clean_Acc=([\d.]+)_NN=(\d+).pdf")

# === MAIN SCRIPT ===
rows = []

for folder_path, subdirs, files in os.walk(root_dir):
    folder_name = os.path.basename(folder_path)

    folder_match = folder_pattern.match(folder_name)
    if not folder_match:
        continue  # Skip folders that don't match the expected pattern

    k, reg_m, reg = folder_match.groups()

    for file in files:
        file_match = file_pattern.match(file)
        if not file_match:
            continue  # Skip files that don't match the expected pattern

        acc, nn = file_match.groups()

        row = {
            "type": "Accuracy",
            "data_set": "salinasA",
            "num_atoms": int(k),
            "reg_m": int(reg_m),
            "reg": float(reg),
            "nn": int(nn),
            "n_clusters": 6,
            "score": float(acc)
        }

        rows.append(row)

# === SAVE TO EXCEL ===
new_df = pd.DataFrame(rows)

if excel_file.exists():
    # Load existing and append only new rows
    old_df = pd.read_excel(excel_file)
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    combined_df = new_df

# Remove duplicates just in case (optional)
combined_df.drop_duplicates(inplace=True)

# Save the updated Excel file
combined_df.to_excel(excel_file, index=False)

print(f"✓ Extraction complete. {len(new_df)} new rows added.")
