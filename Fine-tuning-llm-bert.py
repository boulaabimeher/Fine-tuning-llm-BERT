# ============================================================
# BERT Fine-Tuning for Multi-Class Sentiment Analysis
# ============================================================

# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

# -----------------------------
# 2. Load Dataset
# -----------------------------
DATA_PATH = "twitter_multi_class_sentiment.csv"
df = pd.read_csv(DATA_PATH)

# Basic inspection
df.info()
print(df.isnull().sum())
print(df["label"].value_counts())

# -----------------------------
# 3. Dataset Analysis
# -----------------------------
# Class distribution
df["label_name"].value_counts(ascending=True).plot.barh(
    title="Frequency of Sentiment Classes"
)
plt.show()

# Words per tweet
df["words_per_tweet"] = df["text"].str.split().apply(len)
df.boxplot(column="words_per_tweet", by="label_name", figsize=(8, 5))
plt.title("Tweet Length Distribution by Class")
plt.suptitle("")
plt.ylabel("Number of Words")
plt.show()

# -----------------------------
# 4. Tokenizer
# -----------------------------
MODEL_CHECKPOINT = "../bert-base-uncased-local"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Tokenizer sanity check
sample_text = "I love machine learning! Tokenization is awesome!!"
print(tokenizer(sample_text))
print("Vocab size:", tokenizer.vocab_size)
print("Max length:", tokenizer.model_max_length)

# -----------------------------
# 5. Train / Validation / Test Split
# -----------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_name"], random_state=42
)

test_df, val_df = train_test_split(
    temp_df, test_size=1 / 3, stratify=temp_df["label_name"], random_state=42
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# -----------------------------
# 6. HuggingFace Dataset
# -----------------------------
dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    }
)


# -----------------------------
# 7. Tokenization Function
# -----------------------------
def tokenize(batch):
    """
    Tokenize tweet text using BERT tokenizer.
    """
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
    )


emotion_encoded = dataset.map(tokenize, batched=True)

# -----------------------------
# 8. Label Mapping
# -----------------------------
label2id = {row["label_name"]: row["label"] for row in dataset["train"]}
id2label = {v: k for k, v in label2id.items()}

print("Label2ID:", label2id)

# -----------------------------
# 9. Model Initialization
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, config=config
).to(device)

# -----------------------------
# 10. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="bert_base_train_dir",
    overwrite_output_dir=True,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=100,
    save_strategy="epoch",
    disable_tqdm=False,
)


# -----------------------------
# 11. Metrics
# -----------------------------
def compute_metrics(pred):
    """
    Compute Accuracy and Weighted F1-score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# -----------------------------
# 12. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotion_encoded["train"],
    eval_dataset=emotion_encoded["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# 13. Evaluation
# -----------------------------
preds_output = trainer.predict(emotion_encoded["test"])

y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = emotion_encoded["test"]["label"]

print(classification_report(y_true, y_pred))

# -----------------------------
# 14. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=label2id.keys(),
    yticklabels=label2id.keys(),
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# -----------------------------
# 15. Prediction Function
# -----------------------------
def predict_sentiment(text):
    """
    Predict sentiment label for a single input text.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred_id]


# Example
example_text = "I am super happy today. I got it done. Finally!!"
print("Prediction:", predict_sentiment(example_text))

# -----------------------------
# 16. Save Model & Pipeline
# -----------------------------
trainer.save_model("bert-base-uncased-sentiment-model")

classifier = pipeline(
    "text-classification",
    model="bert-base-uncased-sentiment-model",
    tokenizer=tokenizer,
)

print(classifier([example_text, "hello, how are you?", "love you", "i am feeling low"]))
