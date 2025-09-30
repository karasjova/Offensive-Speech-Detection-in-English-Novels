import os
import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from collections import Counter
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util

from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, AutoConfig
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax


device = 0 if torch.cuda.is_available() else -1
print("Using device:", "CUDA" if device==0 else "CPU")

nltk.download("punkt")
nltk.download("vader_lexicon")
vader = SentimentIntensityAnalyzer()

# ========================
# Paths
# ========================
book_path = "/kaggle/input/last-exit-to-brooklyn/last-exit-to-brooklyn.txt"
hurtlex_path = "/kaggle/input/hurtkex-filtered/hurtlex_filtered.xlsx"
labelled_path = "/kaggle/input/new-salinger-labelled/new_salinger_labelled.csv"
matched_csv = "matched_sentences_hatebert.csv"
finetuned_model_dir = "./hatebert-finetuned"
batch_size_pred = 16
similarity_threshold = 0.6

# ========================
# Step 1: Load & tokenize text
# ========================

def preprocess_text(text: str, remove_nonprintable: bool = True) -> str:
    text = re.sub(r'\r?\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    if remove_nonprintable:
        text = ''.join(c for c in text if c.isprintable())
    text = text.strip()
    return text
    
with open(book_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

text = preprocess_text(raw_text)
    
def tokenize_text(text: str, lower: bool=True, only_alpha: bool=True):
    if lower:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    if only_alpha:
        tokens = [t for t in tokens if t.isalpha()]
    freq_dict = Counter(tokens)
    return list(freq_dict.keys()), freq_dict

tokens, freq_dict = tokenize_text(text)
print(f"Unique tokens count: {len(tokens)}")

# ========================
# Step 2: Load HurtLex
# ========================
hurtlex_df = pd.read_excel(hurtlex_path)
hurtlex_words = set(hurtlex_df["lemma"].dropna().str.lower().unique())
print(f"Unique HurtLex words count: {len(hurtlex_words)}")
hurtlex_list = list(hurtlex_words)

# ========================
# Step 3: Expand lexicon
# ========================
stemmer = PorterStemmer()
model_st = SentenceTransformer('all-MiniLM-L6-v2')

text_embeddings = model_st.encode(tokens, convert_to_tensor=True)
hurtlex_embeddings = model_st.encode(hurtlex_list, convert_to_tensor=True)

results = []

for i, token in enumerate(tokens):
    token_emb = text_embeddings[i]
    cosine_scores = util.cos_sim(token_emb, hurtlex_embeddings)[0]
    max_score = float(cosine_scores.max())
    matched_idx = int(torch.argmax(cosine_scores))
    matched_word = hurtlex_list[matched_idx]

    if max_score < similarity_threshold:
        continue
    if token == matched_word:
        continue

    # Polarity
    vader_compound = vader.polarity_scores(token)["compound"]
    tb_polarity = TextBlob(token).sentiment.polarity
    level_of_agreement = int((vader_compound < 0) and (tb_polarity < 0))

    token_type = "extra_form" if stemmer.stem(token)==stemmer.stem(matched_word) else "new_lemma"

    results.append({
        "token": token,
        "matched_word": matched_word,
        "similarity": max_score,
        "vader_compound": vader_compound,
        "textblob_polarity": tb_polarity,
        "level_of_agreement": level_of_agreement,
        "type": token_type
    })

df_results = pd.DataFrame(results)
neg_tokens_set = set(df_results["token"].unique())
df_neg_tokens = pd.DataFrame({"word": list(neg_tokens_set)})
df_neg_tokens.to_csv("neg_tokens_set.csv", index=False, encoding="utf-8-sig")
print(f"Negative tokens: {len(neg_tokens_set)}")
all_words_set = neg_tokens_set.union(hurtlex_words)
print(f"All words in expanded lexicon: {len(all_words_set)}")

# ========================
# Step 4: Preprocess text into sentences
# ========================
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

doc = nlp(text)
sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ========================
# Step 5: Filter sentences by expanded lexicon
# ========================
matched_sentences = []
for sent in sentences:
    sent_lower = sent.lower()
    if any(word in sent_lower for word in all_words_set):
        matched_sentences.append({"sentence": sent})

df_matched = pd.DataFrame(matched_sentences)
df_matched.to_csv(matched_csv, sep=";", index=False, encoding="utf-8-sig")
print(f"{matched_csv} saved ({len(df_matched)} sentences matched)")

# ========================
# Step 6: Fine-tune HateBERT
# ========================
df_labelled = pd.read_csv(labelled_path, sep=";")
df_labelled.columns = df_labelled.columns.str.strip()
label_map = {'offensive': 1, 'not_offensive': 0}
df_labelled['label'] = df_labelled['offensive'].map(label_map)
df_labelled['input_text'] = df_labelled['sentence']
df_hatebert = df_labelled[['input_text', 'label']]

train_df, test_df = train_test_split(df_hatebert, test_size=0.2, stratify=df_hatebert['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2
config.problem_type = "single_label_classification"
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

def tokenize(batch):
    return tokenizer(batch["input_text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.remove_columns(["input_text", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["input_text", "__index_level_0__"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir=finetuned_model_dir,
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# ========================
# Step 7: Predictions on matched sentences
# ========================
hatebert_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device
)

predicted_labels = []
predicted_scores = []

for i in range(0, len(df_matched), batch_size_pred):
    batch_texts = df_matched["sentence"].iloc[i:i+batch_size_pred].tolist()
    batch_texts = [t[:512] for t in batch_texts]
    batch_results = hatebert_pipeline(batch_texts, truncation=True, max_length=512)
    for res in batch_results:
        predicted_labels.append(res["label"])
        predicted_scores.append(res["score"])

df_matched["hatebert_label"] = predicted_labels
df_matched["hatebert_score"] = predicted_scores
df_matched.to_csv("matched_sentences_hatebert_predicted.csv", index=False, sep=";", encoding="utf-8-sig")
print("matched_sentences_hatebert_predicted.csv saved")

# ========================
# Plotting
# ========================
sns.countplot(x="hatebert_label", data=df_matched)
plt.title("Class distribution of predicted offensive sentences")
plt.show()

plt.hist(df_matched["hatebert_score"], bins=20, color='orange', edgecolor='black')
plt.title("Histogram of predicted offensive probabilities")
plt.xlabel("Probability (offensive)")
plt.ylabel("Number of samples")
plt.grid(True)
plt.show()
