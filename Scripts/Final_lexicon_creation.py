import nltk
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('punkt')
nltk.download('vader_lexicon')

# ====================
# Tokenization function
# ====================
def tokenize_text(text: str, lower: bool = True, only_alpha: bool = True):

    if lower:
        text = text.lower()

    tokens = nltk.word_tokenize(text)

    if only_alpha:
        tokens = [t for t in tokens if t.isalpha()]

    freq_dict = Counter(tokens)

    unique_tokens = list(freq_dict.keys())

    return unique_tokens, freq_dict

# ====================
# Input files and parameters
# ====================
salinger_book_path = "/kaggle/input/salinger2/Salinger - The Catcher in the Rye -English.txt"
hurtlex_path = "/kaggle/input/hurtkex-filtered/hurtlex_filtered.xlsx"
threshold = 0.6

# ====================
# Load and tokenize text
# ====================
with open(salinger_book_path, "r", encoding="utf-8") as f:
    text = f.read()

tokens, freq_dict = tokenize_text(text)

print(f"Unique tokens count: {len(tokens)}")

# ====================
# Load HurtLex
# ====================
hurtlex_df = pd.read_excel(hurtlex_path)
hurtlex_words = set(hurtlex_df["lemma"].dropna().str.lower().unique())
print(f"Unique Hurtlex words count: {len(hurtlex_words)}")
hurtlex_list = list(hurtlex_words)

# ====================
# Initialize models and tools
# ====================
stemmer = PorterStemmer()
analyzer = SentimentIntensityAnalyzer()
model = SentenceTransformer('all-MiniLM-L6-v2')

# ====================
# Vectorization
# ====================
text_embeddings = model.encode(tokens, convert_to_tensor=True)
hurtlex_embeddings = model.encode(hurtlex_list, convert_to_tensor=True)

# ====================
# Search for new words
# ====================
results = []

for i, token in enumerate(tokens):
    token_emb = text_embeddings[i]
    cosine_scores = util.cos_sim(token_emb, hurtlex_embeddings)[0]

    max_score = float(cosine_scores.max())
    matched_idx = int(torch.argmax(cosine_scores))
    matched_word = hurtlex_list[matched_idx]

    #Cosine similarity threshold
    if max_score < threshold:
        continue

    #Skip exact matches with HurtLex
    if token == matched_word:
        continue

    #VADER polarity
    vader_compound = analyzer.polarity_scores(token)["compound"]
    vader_neg = vader_compound < 0

    #TextBlob polarity
    tb_polarity = TextBlob(token).sentiment.polarity
    tb_neg = tb_polarity < 0

    #Agreement level
    level_of_agreement = int(vader_neg and tb_neg)

    #Token type detection
    if stemmer.stem(token) == stemmer.stem(matched_word):
        token_type = "extra_form"
    else:
        token_type = "new_lemma"

    #Save result 
    results.append({
        "token": token,
        "matched_word": matched_word,
        "similarity": f"{max_score:.4f}",
        "vader_compound": f"{vader_compound:.4f}",
        "textblob_polarity": f"{tb_polarity:.4f}",
        "vader_neg": vader_neg,
        "textblob_neg": tb_neg,
        "level_of_agreement": level_of_agreement,
        "type": token_type
    })
# ====================
# Save token comparison
# ====================
df_results = pd.DataFrame(results)
df_results.to_csv("tokens_comparison_vader_textblob.csv", index=False, encoding="utf-8")

# ====================
# Collect negative tokens by VAder
# ====================
neg_tokens_set = set(df_results[df_results["vader_neg"]]["token"].unique())
print(f"Unique negative tokens count: {len(neg_tokens_set)}")  # должно быть 114

# ====================
# Merge with HurtLex words
# ====================
all_words_set = neg_tokens_set.union(hurtlex_words)
print(f"All words count: {len(all_words_set)}") 

df_all_words = pd.DataFrame({"word": list(all_words_set)})
df_all_words.to_csv("all_hurtlex_and_neg_vader_words.csv", index=False, encoding="utf-8")

# ====================
# Summary table
# ====================
summary_table = pd.DataFrame({
    "Method": ["VADER", "TextBlob", "Agreement (both)"],
    "Extra_forms": [
        df_results[df_results["vader_neg"] & (df_results["type"]=="extra_form")].shape[0],
        df_results[df_results["textblob_neg"] & (df_results["type"]=="extra_form")].shape[0],
        df_results[(df_results["level_of_agreement"]==1) & (df_results["type"]=="extra_form")].shape[0]
    ],
    "New_lemmas": [
        df_results[df_results["vader_neg"] & (df_results["type"]=="new_lemma")].shape[0],
        df_results[df_results["textblob_neg"] & (df_results["type"]=="new_lemma")].shape[0],
        df_results[(df_results["level_of_agreement"]==1) & (df_results["type"]=="new_lemma")].shape[0]
    ]
})
summary_table["Total"] = summary_table["Extra_forms"] + summary_table["New_lemmas"]
print(summary_table)

# ====================
# Plot comparison diagram
# ====================
labels = ["Extra forms", "New lemmas"]
vader_counts = [summary_table.loc[0, "Extra_forms"], summary_table.loc[0, "New_lemmas"]]
tb_counts = [summary_table.loc[1, "Extra_forms"], summary_table.loc[1, "New_lemmas"]]
agreement_counts = [summary_table.loc[2, "Extra_forms"], summary_table.loc[2, "New_lemmas"]]

x = range(len(labels))
width = 0.25

plt.figure(figsize=(8,6))
bars_vader = plt.bar([p - width for p in x], vader_counts, width=width, color='red', label='VADER')
bars_tb = plt.bar(x, tb_counts, width=width, color='blue', label='TextBlob')
bars_agree = plt.bar([p + width for p in x], agreement_counts, width=width, color='green', label='Agreement (both)')

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height), ha='center', va='bottom', fontsize=10)

add_labels(bars_vader)
add_labels(bars_tb)
add_labels(bars_agree)

plt.xticks(x, labels)
plt.ylabel("Count of negative words")
plt.title("Comparison of negative words detection methods: VADER vs TextBlob")
plt.legend()
plt.show()
