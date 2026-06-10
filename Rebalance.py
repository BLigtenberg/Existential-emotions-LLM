import pandas as pd
import numpy as np

# Load original files
print("Loading original CSV files...")
train_df = pd.read_csv("trainingdataset_clean.csv", sep=";")
val_df = pd.read_csv("validationdataset_clean.csv", sep=";")
test_df = pd.read_csv("testdataset_clean.csv", sep=";")

# Pool all data
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# List of label columns
label_columns = [
    "Death Anxiety",
    "Death Acceptance",
    "Loneliness",
    "Solitude",
    "Identity Confusion",
    "Identity Synthesis",
    "Freedom Paralysis",
    "Freedom Responsibility",
    "Meaninglessness",
    "Engagement"
]

# Target capacities: Train (400), Val (50), Test (50)
np.random.seed(42)

all_indices = list(range(len(combined_df)))
assigned_train = []
assigned_val = []
assigned_test = []

def assign_to_split(idx):
    train_cap, val_cap, test_cap = 400, 50, 50
    len_t, len_v, len_te = len(assigned_train), len(assigned_val), len(assigned_test)
    
    scores = []
    if len_t < train_cap:
        scores.append(("train", len_t / train_cap))
    if len_v < val_cap:
        scores.append(("val", len_v / val_cap))
    if len_te < test_cap:
        scores.append(("test", len_te / test_cap))
        
    scores.sort(key=lambda x: x[1])
    chosen_split = scores[0][0]
    
    if chosen_split == "train":
        assigned_train.append(idx)
    elif chosen_split == "val":
        assigned_val.append(idx)
    else:
        assigned_test.append(idx)

# Find positive frequencies of each label across the entire dataset
label_counts = combined_df[label_columns].sum().to_dict()
sorted_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x])

print("Labels sorted by frequency (rarest first):", [(l, label_counts[l]) for l in sorted_labels])

# Assign rows with positive labels first, starting from the rarest label
already_assigned = set()
for label in sorted_labels:
    matching_indices = combined_df[(combined_df[label] == 1) & (~combined_df.index.isin(already_assigned))].index.tolist()
    np.random.shuffle(matching_indices)
    for idx in matching_indices:
        assign_to_split(idx)
        already_assigned.add(idx)

# Assign remaining all-zero rows
unassigned_zeros = [idx for idx in all_indices if idx not in already_assigned]
np.random.shuffle(unassigned_zeros)
for idx in unassigned_zeros:
    assign_to_split(idx)
    already_assigned.add(idx)

print(f"\nFinal split sizes: Train={len(assigned_train)}, Val={len(assigned_val)}, Test={len(assigned_test)}")

# Create dataframes
train_rebalanced = combined_df.iloc[assigned_train].reset_index(drop=True)
val_rebalanced = combined_df.iloc[assigned_val].reset_index(drop=True)
test_rebalanced = combined_df.iloc[assigned_test].reset_index(drop=True)

# Print label distributions across splits
print("\n--- Label Distribution Across New Splits ---")
summary_df = pd.DataFrame({
    "Total Count": [label_counts[l] for l in label_columns],
    "Train": train_rebalanced[label_columns].sum(),
    "Val": val_rebalanced[label_columns].sum(),
    "Test": test_rebalanced[label_columns].sum()
})
print(summary_df)

# Save files
train_rebalanced.to_csv("training_rebalanced.csv", sep=";", index=False)
val_rebalanced.to_csv("validation_rebalanced.csv", sep=";", index=False)
test_rebalanced.to_csv("test_rebalanced.csv", sep=";", index=False)
print("\nRebalanced CSV files saved successfully.")