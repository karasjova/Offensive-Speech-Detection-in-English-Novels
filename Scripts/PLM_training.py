import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# ====================
# Paths
# ====================
DATA_PATH = "/kaggle/input/new-salinger-labelled/new_salinger_labelled.csv"
OUT_DIR = "experiments_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ====================
# Load dataset
# ====================
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()
label_map = {'offensive': 1, 'not_offensive': 0}
df['label'] = df['offensive'].map(label_map)

# ====================
# Scenarios
# ====================
df['sentence_only'] = df['sentence']
df['sentence_plus_first_word'] = df['sentence'] + " " + df['word']
df['sentence_plus_all_words'] = df.groupby('sentence')['word'].transform(lambda x: ' '.join(x))
df['sentence_plus_all_words'] = df['sentence'] + " " + df['sentence_plus_all_words']
scenarios = ['sentence_only', 'sentence_plus_first_word', 'sentence_plus_all_words']

device = 0 if torch.cuda.is_available() else -1

plm_results = []

# ====================
# Zero-shot setting
# ====================

plm_zero_shot_models = {
    "ZeroShot-BART": "facebook/bart-large-mnli",
    "ZeroShot-HateBERT": "GroNLP/hateBERT",
    "ZeroShot-HateXplain": "Hate-speech-CNERG/bert-base-uncased-hatexplain"
}

for model_name, model_path in plm_zero_shot_models.items():
    if model_name == "ZeroShot-BART":
        classifier = pipeline(
            "zero-shot-classification",
            model=model_path,
            device=device
        )
        candidate_labels = ["offensive", "not_offensive"]

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)  
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    for scenario in scenarios:
        y_true = df['label'].tolist()
        y_pred = []

        for sent in df[scenario].tolist():
            if model_name == "ZeroShot-BART":
                res = classifier(sent, candidate_labels=candidate_labels)
                pred = 1 if res['labels'][0] == 'offensive' else 0

            elif model_name == "ZeroShot-HateBERT":
                res = classifier(sent)
                # LABEL_0 / LABEL_1 → преобразуем в 0/1
                pred = 1 if res[0]["label"] == "LABEL_1" else 0

            else:  # ZeroShot-HateXplain
                res = classifier(sent)
                label_upper = res[0]["label"].upper()
                pred = 1 if ("HATE" in label_upper) or ("OFF" in label_upper) else 0

            y_pred.append(pred)

        f1 = f1_score(y_true, y_pred)
        plm_results.append({
            "model": model_name,
            "scenario": scenario,
            "split": "zero-shot",
            "mean_f1": f1
        })
        print(f"{model_name} | {scenario} | F1: {f1:.3f}")

# ====================
# Fine-tuned models 80-20 split
# ====================
ft_models = {
    "HateBERT": "GroNLP/hateBERT",
    "HateXplain": "Hate-speech-CNERG/bert-base-uncased-hatexplain"
}

for model_name, model_path in ft_models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)

    for scenario in scenarios:
        temp_df = df[[scenario, 'label']].rename(columns={scenario:'input_text'})
        train_df, test_df = train_test_split(temp_df, test_size=0.2, stratify=temp_df['label'], random_state=42)

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        def tokenize(batch):
            return tokenizer(batch["input_text"], truncation=True, padding=True)
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
        train_dataset = train_dataset.remove_columns(["input_text", "__index_level_0__"])
        test_dataset = test_dataset.remove_columns(["input_text", "__index_level_0__"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir=f"./{model_name}-{scenario}-80-20",
            eval_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
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
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        f1 = f1_score(predictions.label_ids, y_pred)

        plm_results.append({
            "model": model_name+"-FT",
            "scenario": scenario,
            "split": "80-20",
            "mean_f1": f1
        })

# ====================
# Fine-tuned models 5-fold CV
# ====================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for model_name, model_path in ft_models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for scenario in scenarios:
        temp_df = df[[scenario, 'label']].rename(columns={scenario:'input_text'})
        X = temp_df['input_text'].tolist()
        y = temp_df['label'].tolist()
        f1_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            train_dataset = Dataset.from_pandas(pd.DataFrame({'input_text': X_train, 'label': y_train}))
            test_dataset = Dataset.from_pandas(pd.DataFrame({'input_text': X_test, 'label': y_test}))

            train_dataset = train_dataset.map(tokenize, batched=True)
            test_dataset = test_dataset.map(tokenize, batched=True)
            train_dataset = train_dataset.remove_columns(["input_text"])
            test_dataset = test_dataset.remove_columns(["input_text"])
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            training_args = TrainingArguments(
                output_dir=f"./{model_name}-{scenario}-fold{fold_idx}",
                eval_strategy="epoch",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                logging_dir="./logs",
                logging_steps=10,
                save_strategy="no",
                report_to=[]
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=2, ignore_mismatched_sizes=True
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
            predictions = trainer.predict(test_dataset)
            logits = predictions.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu().numpy()
            y_pred = np.argmax(logits, axis=1)
            f1_scores.append(f1_score(y_test, y_pred))

        mean_f1 = np.mean(f1_scores)
        plm_results.append({
            "model": model_name+"-FT",
            "scenario": scenario,
            "split": "5-fold CV",
            "mean_f1": mean_f1
        })

# ====================
# Save results
# ====================
df_results = pd.DataFrame(plm_results)
csv_path = os.path.join(OUT_DIR, "plm_results_combined.csv")
df_results.to_csv(csv_path, index=False)
print("Results saved to CSV:", csv_path)
