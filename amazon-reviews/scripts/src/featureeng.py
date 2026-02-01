###########################################################
### step 2 - feature engineering (OPTIMIZED)
###########################################################

# =========================
# Libraries
# =========================
import time
import gc
import string
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

# =========================
# Initial setup
# =========================
start_time = time.time()
print("date..............:", datetime.now())

# =========================
# Load dataset
# =========================
print("Loading dataset...")
df = pd.read_feather("../datasets/feather/cleaned.ftr")

percentage = (
    df["Score"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename("percentage")
)

print(percentage.map(lambda x: f"{x:.2f}%"))


# =========================
# Exploratory analysis
# =========================
ax = df["Score"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(10, 5),
    title="Contagem de Reviews por Estrelas"
)
ax.set_xlabel("Review Stars")
ax.set_ylabel("Contagem")
plt.tight_layout()
plt.savefig("../pngs/counting_reviews_stars.png")
plt.close()

# =========================
# Text preprocessing (EDA only)
# =========================
print("Generating word cloud...")

stop_pt = stopwords.words("portuguese")
stop_en = stopwords.words("english")
stop_all = set(stop_pt + stop_en)

texts = " ".join(df["Text"].astype(str)).lower()
tokens = texts.split()
tokens = [t.strip(string.punctuation) for t in tokens]
tokens = [t for t in tokens if t and t not in stop_all]

freqdist = Counter(tokens)

cleaned_text = " ".join(tokens)
wordcloud = WordCloud(width=800, height=800, background_color="white").generate(cleaned_text)

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("../pngs/word_clouds.png")
plt.close()

# =========================
# Save outputs
# =========================
print("Saving feature-engineered datasets...")

df.reset_index(drop=True, inplace=True)
df.to_feather("../datasets/feather/featured.ftr")

# =========================
# Final logs
# =========================
time_exec_min = round((time.time() - start_time) / 60, 4)

print(f"Execution time: {time_exec_min} minutes")
print("Feature engineering completed successfully.")
print("Next step: modeling.")