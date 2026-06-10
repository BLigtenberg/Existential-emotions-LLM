import pandas as pd
import os
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit.model_card import SetFitModelCardData

# Patch SetFitModelCardData to only pass parameters that have init=True
original_init = SetFitModelCardData.__init__
def patched_init(self, *args, **kwargs):
    valid_fields = [k for k, field in self.__dataclass_fields__.items() if field.init]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    original_init(self, *args, **filtered_kwargs)

SetFitModelCardData.__init__ = patched_init

print("--- Step 1: Loading Rebalanced Datasets ---")
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

train_df = pd.read_csv("training_rebalanced.csv", sep=";")
val_df = pd.read_csv("validation_rebalanced.csv", sep=";")
test_df = pd.read_csv("test_rebalanced.csv", sep=";")

train_dataset = Dataset.from_dict({
    "text": train_df["Text"].tolist(),
    "label": train_df[label_columns].values.tolist()
})

val_dataset = Dataset.from_dict({
    "text": val_df["Text"].tolist(),
    "label": val_df[label_columns].values.tolist()
})

test_dataset = Dataset.from_dict({
    "text": test_df["Text"].tolist(),
    "label": test_df[label_columns].values.tolist()
})

print(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test examples.")

print("\n--- Step 2: Loading SetFit Model (all-MiniLM-L6-v2) ---")
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    multi_target_strategy="one-vs-rest",
)

print("\n--- Step 3: Setting Up Trainer ---")
# Using 5 epochs and 2 iterations to match original notebook settings for quick CPU training (~38 minutes)
args = TrainingArguments(
    batch_size=8,
    num_epochs=5,
    num_iterations=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n--- Step 4: Starting Full SetFit Training ---")
print("Fine-tuning the sentence embedding space (Stage 1) and training the classifier head (Stage 2)...")
trainer.train()

print("\n--- Step 5: Evaluating Model ---")
eval_results = trainer.evaluate(test_dataset)
print("Evaluation results on test dataset:", eval_results)

print("\n--- Step 6: Saving Model ---")
output_dir = "setfit-minilm-existential-emotions"
model.save_pretrained(output_dir)
print(f"Model saved successfully in '{output_dir}' directory!")