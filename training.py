# train.py
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
df = pd.read_csv('recipes.csv')

# Encode user and recipe IDs
user_encoder = LabelEncoder()
recipe_encoder = LabelEncoder()

df['user_enc'] = user_encoder.fit_transform(df['user_id'])
df['recipe_enc'] = recipe_encoder.fit_transform(df['recipe_id'])

# Create pivot table
pivot = df.pivot_table(index='user_enc', columns='recipe_enc', values='rating').fillna(0)

# Convert to sparse matrix
sparse_matrix = csr_matrix(pivot.values)

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'pivot': pivot,
        'user_encoder': user_encoder,
        'recipe_encoder': recipe_encoder,
        'df': df
    }, f)

print("✅ Model trained and saved to model.pkl")
