# ==========================================
# Movie Recommendation using Apriori
# Optimized Version (Fast + Clean)
# ==========================================

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

# ------------------------------------------
# STEP 0: Check files
# ------------------------------------------
print("Files in directory:", os.listdir())

# ------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------

ratings = pd.read_csv(
    'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

movies = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1],
    names=['movie_id', 'title']
)

# Merge datasets
data = pd.merge(ratings, movies, on='movie_id')

print("\nDataset Loaded Successfully!")

# ------------------------------------------
# STEP 2: Preprocessing
# ------------------------------------------

# Convert ratings to liked (1/0)
data['liked'] = data['rating'] >= 4

# Keep only liked movies
liked_movies = data[data['liked'] == True]

# ------------------------------------------
# STEP 3: Create User-Movie Matrix
# ------------------------------------------

basket = liked_movies.groupby(['user_id', 'title'])['liked'] \
    .sum().unstack().fillna(0)

# Convert to boolean (IMPORTANT for speed)
basket = basket.astype(bool)

print("\nOriginal Matrix Shape:", basket.shape)

# ------------------------------------------
# STEP 4: Reduce Dataset (STRONG FILTER)
# ------------------------------------------

# Keep only very popular movies (liked by at least 100 users)
movie_counts = liked_movies['title'].value_counts()
popular_movies = movie_counts[movie_counts > 100].index

basket = basket[popular_movies]

print("Reduced Matrix Shape:", basket.shape)

# ------------------------------------------
# STEP 5: Apply Apriori (SAFE SETTINGS)
# ------------------------------------------

print("\nRunning Apriori...")

frequent_itemsets = apriori(
    basket,
    min_support=0.08,     # increase support
    use_colnames=True,
    max_len=2             # IMPORTANT: only pairs (prevents explosion)
)

print("Frequent Itemsets Found:", len(frequent_itemsets))

# ------------------------------------------
# STEP 6: Generate Association Rules
# ------------------------------------------

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# Sort by lift (best relationships first)
rules = rules.sort_values(by='lift', ascending=False)

# ------------------------------------------
# STEP 7: Display Results
# ------------------------------------------

print("\nTop Association Rules:\n")

if len(rules) == 0:
    print("No rules found. Try lowering support/confidence.")
else:
    for i, row in rules.head(10).iterrows():
        print(f"Rule: {set(row['antecedents'])} -> {set(row['consequents'])}")
        print(f"Support: {row['support']:.3f}")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Lift: {row['lift']:.3f}")
        print("-" * 50)

# ------------------------------------------
# STEP 8: Save Results (for report)
# ------------------------------------------

rules.to_csv("association_rules.csv", index=False)

print("\nResults saved to association_rules.csv")