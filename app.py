# app.py
import streamlit as st
import pickle
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
pivot = data['pivot']
user_encoder = data['user_encoder']
recipe_encoder = data['recipe_encoder']
df = data['df']

st.title("🍽️ Food Recipe Recommendation System")

user_input = st.text_input("Enter your User ID (e.g., U001):")

if user_input:
    try:
        user_index = user_encoder.transform([user_input])[0]
        user_vector = pivot.loc[user_index].values.reshape(1, -1)

        # Get nearest users
        distances, indices = model.kneighbors(user_vector, n_neighbors=3)

        # Get recipes rated by similar users
        similar_user_indices = pivot.index[indices.flatten()[1:]]
        recommendations = []

        for sim_user in similar_user_indices:
            rated = df[df['user_enc'] == sim_user]
            for _, row in rated.iterrows():
                if row['recipe_id'] not in df[df['user_id'] == user_input]['recipe_id'].values:
                    recommendations.append((row['recipe_name'], row['ingredients']))

        if recommendations:
            st.subheader("🍲 Recommended Recipes:")
            shown = set()
            for name, ing in recommendations:
                if name not in shown:
                    st.markdown(f"**🍛 {name}**")
                    st.markdown(f"📝 *Ingredients:* {ing}")
                    st.markdown("---")
                    shown.add(name)
        else:
            st.info("No new recommendations found. Try with another user.")
    except:
        st.error("Invalid User ID. Try again.")
