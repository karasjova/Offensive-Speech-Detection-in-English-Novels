import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ====================
# Paths
# ====================
DATA_CSV = "/kaggle/input/new-salinger-labelled/new_salinger_labelled.csv"
OUT_DIR = "experiments_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ====================
# Load data
# ====================
df = pd.read_csv(DATA_CSV, sep=";")
df.columns = df.columns.str.strip()
label_map = {'offensive': 1, 'not_offensive': 0}
df['label'] = df['offensive'].map(label_map)

# ====================
# Feature scenarios
# ====================
df['sentence_only'] = df['sentence']
df['sentence_plus_first_word'] = df['sentence'] + " " + df['word']
df['sentence_plus_all_words'] = df.groupby('sentence')['word'].transform(lambda x: ' '.join(x))
df['sentence_plus_all_words'] = df['sentence'] + " " + df['sentence_plus_all_words']
scenarios = ['sentence_only', 'sentence_plus_first_word', 'sentence_plus_all_words']

# ====================
# Models
# ====================
models = [
    ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("SVM", LinearSVC(random_state=42)),
    ("NaiveBayes", MultinomialNB())
]

# ====================
# Train/Test function
# ====================
def run_train_test():
    results = []
    for model_name, model in models:
        for scenario in scenarios:
            X = df[scenario]
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X_train_vec = vec.fit_transform(X_train)
            X_test_vec = vec.transform(X_test)

            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            f1 = f1_score(y_test, y_pred)
            results.append({"model": model_name, "scenario": scenario, "f1": f1})
            print(f"➡️ Train/Test F1 for {model_name} ({scenario}): {f1:.4f}")

    df_results = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x="model", y="f1", hue="scenario", data=df_results)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height()+0.005, f"{p.get_height():.2f}", ha="center")
    plt.title("F1-score (Train/Test Split)")
    plt.ylabel("F1-score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_train_test.png"))
    plt.show()

    df_results.to_csv(os.path.join(OUT_DIR, "f1_train_test.csv"), index=False)
    return df_results

# ====================
# Cross-validation function
# ====================
def run_cross_validation():
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models:
        for scenario in scenarios:
            X = df[scenario].values
            y = df['label'].values
            fold_f1 = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                X_train_vec = vec.fit_transform(X_train)
                X_test_vec = vec.transform(X_test)

                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                f1 = f1_score(y_test, y_pred)
                fold_f1.append(f1)

            mean_f1 = sum(fold_f1)/len(fold_f1)
            results.append({"model": model_name, "scenario": scenario, "mean_f1": mean_f1})
            print(f"➡️ 5-Fold CV Mean F1 for {model_name} ({scenario}): {mean_f1:.4f}")

    df_cv = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x="model", y="mean_f1", hue="scenario", data=df_cv)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height()+0.005, f"{p.get_height():.2f}", ha="center")
    plt.title("Mean F1-score (5-Fold Cross-validation)")
    plt.ylabel("Mean F1-score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_cross_validation.png"))
    plt.show()

    df_cv.to_csv(os.path.join(OUT_DIR, "f1_cross_validation.csv"), index=False)
    return df_cv

# ====================
# Run experiments
# ====================
print("Running Train/Test experiment...")
df_train_test = run_train_test()

print("\n Running Cross-validation experiment...")
df_cross_val = run_cross_validation()

print("\n All experiments finished. Results saved in:", OUT_DIR)
