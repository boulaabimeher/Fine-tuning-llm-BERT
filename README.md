# BERT Fine-Tuning for Multi-Class Sentiment Analysis

This project fine-tunes a **BERT-based model** for **multi-class sentiment (emotion) classification** on Twitter data using the Hugging Face 🤗 Transformers library.

The pipeline covers:
- Data analysis and visualization
- Tokenization using BERT
- Train / validation / test split
- Model fine-tuning
- Evaluation with metrics and confusion matrix
- Inference using both custom function and Hugging Face pipeline

---

## 📁 Project Structure

```text
.
├── twitter_multi_class_sentiment.csv
├── train_script.py          # Main training & evaluation script
├── bert_base_train_dir/     # Training outputs & checkpoints
├── bert-base-uncased-sentiment-model/  # Saved fine-tuned model
└── README.md

📊 Dataset

    Source: Twitter multi-class sentiment dataset

    Columns:

        text – Tweet text

        label – Numeric label

        label_name – Human-readable sentiment label

    Task: Multi-class classification (e.g. joy, sadness, anger, fear, etc.)

⚙️ Requirements

Install dependencies using:

pip install -U pandas numpy scikit-learn matplotlib seaborn torch datasets transformers

    ⚠️ For GPU training, ensure that PyTorch with CUDA is installed.

🚀 How to Run

    Clone the repository

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

    Update dataset path (if needed)
    Edit this line in the script:

DATA_PATH = "/05-Fine_Tuning_BERT_for_Sentiment/twitter_multi_class_sentiment.csv"

    Run training

python train_script.py

The model will be trained and saved to:

bert-base-uncased-sentiment-model/

🧠 Model

    Base model: bert-base-uncased

    Framework: Hugging Face Transformers

    Training objective: Multi-class classification

    Loss: Cross-entropy

    Optimizer: AdamW

    Learning rate: 2e-5

    Epochs: 2

    Batch size: 64

📈 Evaluation Metrics

The model is evaluated using:

    Accuracy

    Weighted F1-score

    Classification Report

    Confusion Matrix

Example output:

Accuracy: 0.87
F1-score: 0.86

🔍 Inference
Custom Prediction Function

predict_sentiment("I am super happy today!")

Hugging Face Pipeline

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bert-base-uncased-sentiment-model"
)

classifier([
    "I am very happy today!",
    "I feel sad and tired.",
    "This is terrible."
])

💾 Saving & Loading Model

The trained model is saved automatically using:

trainer.save_model("bert-base-uncased-sentiment-model")

You can reload it later using:

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased-sentiment-model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased-sentiment-model"
)

🧪 Example Predictions
Text	Prediction
"I love this!"	Joy
"This is awful."	Anger
"I feel nervous."	Fear
📌 Notes

    Ensure labels are stratified during splitting.

    Adjust batch size if GPU memory is limited.

    For reproducibility, consider setting random seeds.

🔮 Future Improvements

    Add early stopping

    Hyperparameter tuning

    Model compression / quantization

    Deployment with FastAPI or Streamlit


✨ Author

Meher Boulaabi