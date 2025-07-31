import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="Laptop Recommender",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
# Using @st.cache_data to load and clean data only once, improving performance.
@st.cache_data
def load_and_clean_data(filepath):
    """
    Loads the laptop dataset from a CSV file, cleans it, and prepares it for the app.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the same directory as the script.")
        return None

    # Drop unnecessary column
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # --- Data Cleaning ---
    df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(int)
    df['Ram'] = df['Ram'].str.extract('(\d+)').astype(int)

    def convert_storage_to_gb(storage):
        if pd.isna(storage): return np.nan
        storage = str(storage).lower()
        if 'tb' in storage:
            num = re.findall(r'(\d+\.?\d*)', storage)
            return float(num[0]) * 1024 if num else np.nan
        elif 'gb' in storage:
            num = re.findall(r'(\d+\.?\d*)', storage)
            return float(num[0]) if num else np.nan
        return np.nan

    df['SSD'] = df['SSD'].apply(convert_storage_to_gb)
    df['Brand'] = df['Model'].apply(lambda x: x.split()[0])

    # FIX: Replaced inplace=True to remove FutureWarning
    # Fill missing values for recommendation features
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['SSD'] = df['SSD'].fillna(df['SSD'].median())
    
    return df

# --- Recommendation Logic Caching ---
@st.cache_data
def calculate_similarity(_df_rec):
    """
    Calculates the cosine similarity matrix for the recommendation features.
    The _df_rec argument ensures this function re-runs only if the input data changes.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(_df_rec)
    cosine_sim = cosine_similarity(df_scaled)
    return cosine_sim, scaler

# --- Recommendation Function ---
def recommend_laptops(ram, ssd, budget, cosine_sim, scaler, df, df_rec, top_n=5):
    """
    Recommends top_n laptops based on user input.
    """
    # Create a DataFrame for the user's input
    user_input = pd.DataFrame({
        'Rating': [df['Rating'].mean()],  # Use average rating as a placeholder
        'Ram': [ram],
        'SSD': [ssd]
    })

    # Scale the user's input
    user_input_scaled = scaler.transform(user_input)

    # Calculate similarity scores
    sim_scores = cosine_similarity(user_input_scaled, scaler.transform(df_rec)).flatten()

    # Filter laptops within the user's budget
    budget_friendly_indices = df[df['Price'] <= budget].index
    
    # Get similarity scores for budget-friendly laptops
    filtered_sim_scores = sim_scores[budget_friendly_indices]
    
    # Get the indices of the top N most similar laptops from the filtered list
    # Ensure we don't request more recommendations than available
    num_recommendations = min(top_n, len(filtered_sim_scores))
    if num_recommendations == 0:
        return pd.DataFrame() # Return empty DataFrame if no matches

    top_indices_filtered = filtered_sim_scores.argsort()[-num_recommendations:][::-1]
    
    # Get the original indices from the main DataFrame
    recommended_laptop_indices = budget_friendly_indices[top_indices_filtered]
    
    return df.iloc[recommended_laptop_indices]

# --- Main App UI ---
st.title("💻 Laptop Recommendation Engine")
st.markdown("Find the perfect laptop based on your needs and budget. Adjust the sliders on the left and click the button to get your recommendations!")

# Load data and prepare for recommendation
df = load_and_clean_data('laptop.csv')

if df is not None:
    features_for_rec = ['Rating', 'Ram', 'SSD']
    df_rec = df[features_for_rec].copy()
    cosine_sim, scaler = calculate_similarity(df_rec)

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Your Preferences")

    # Get min/max for sliders from the dataframe
    ram_options = sorted(df['Ram'].unique())
    ssd_options = sorted(df['SSD'].dropna().unique().astype(int))
    min_price, max_price = int(df['Price'].min()), int(df['Price'].max())

    # Create sliders and selectors in the sidebar
    selected_ram = st.sidebar.select_slider(
        "RAM (in GB)",
        options=ram_options,
        value=ram_options[len(ram_options) // 2]  # Default to middle value
    )

    selected_ssd = st.sidebar.select_slider(
        "SSD Storage (in GB)",
        options=ssd_options,
        value=ssd_options[len(ssd_options) // 2] # Default to middle value
    )

    selected_budget = st.sidebar.slider(
        "Maximum Budget (in ₹)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price + max_price) // 2, # Default to middle of price range
        step=1000
    )

    # Recommendation button
    if st.sidebar.button("Find My Laptop", type="primary"):
        recommendations = recommend_laptops(
            selected_ram, selected_ssd, selected_budget,
            cosine_sim, scaler, df, df_rec
        )

        st.subheader("Here are your top recommendations:")

        if recommendations.empty:
            st.warning("No laptops found for your criteria. Try increasing your budget or adjusting the specs.")
        else:
            # Display results in cards
            num_cols = len(recommendations)
            cols = st.columns(num_cols)
            
            for i, row in enumerate(recommendations.itertuples()):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"##### {row.Brand}")
                        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
                        st.image(f"https://placehold.co/400x250/003366/FFFFFF?text={row.Brand}", use_container_width=True)
                        st.markdown(f"**{row.Model}**")
                        st.markdown(f"### **Price: ₹{row.Price:,}**")
                        st.markdown(f"**RAM:** {int(row.Ram)} GB | **SSD:** {int(row.SSD)} GB")
                        st.markdown(f"**Rating:** {row.Rating}/100")
    else:
        st.info("Adjust your preferences in the sidebar and click 'Find My Laptop'.")
