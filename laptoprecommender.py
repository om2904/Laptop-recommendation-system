# Import necessary libraries
import pandas as pd
import numpy as np
import re

# For modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# For recommendation engine
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('laptop.csv')

# Display the first few rows and info
print(df.head())
print(df.info())
# --- Data Cleaning ---

# Drop the 'Unnamed: 0' column as it's just an index
df.drop('Unnamed: 0', axis=1, inplace=True)

# Clean the 'Price' column: remove '₹', commas, and convert to integer
df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(int)

# Clean 'Ram' and 'SSD' columns to extract numerical values
# For RAM, just take the number
df['Ram'] = df['Ram'].str.extract('(\d+)').astype(int)

# For SSD, handle 'GB' and 'TB' and convert all to GB
def convert_storage_to_gb(storage):
    if pd.isna(storage):
        return None
    storage = str(storage).lower()
    if 'tb' in storage:
        # Extract number and multiply by 1024
        num = re.findall(r'(\d+\.?\d*)', storage)
        return float(num[0]) * 1024 if num else None
    elif 'gb' in storage:
        # Extract number
        num = re.findall(r'(\d+\.?\d*)', storage)
        return float(num[0]) if num else None
    return None

df['SSD'] = df['SSD'].apply(convert_storage_to_gb)

# --- Feature Engineering ---

# Extract Brand from the 'Model' column
df['Brand'] = df['Model'].apply(lambda x: x.split()[0])

# Extract Processor Brand (Intel/AMD) from 'Generation'
def get_processor_brand(gen_string):
    if pd.isna(gen_string):
        return 'Other'
    gen_string = str(gen_string).lower()
    if 'intel' in gen_string:
        return 'Intel'
    elif 'amd' in gen_string:
        return 'AMD'
    elif 'apple' in gen_string:
        return 'Apple'
    else:
        return 'Other'

df['Processor Brand'] = df['Generation'].apply(get_processor_brand)

# Fill missing 'Rating' and 'SSD' values (e.g., with the median)
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['SSD'].fillna(df['SSD'].median(), inplace=True)

# Drop original columns that have been processed and are too complex for this model
df_model = df.drop(['Model', 'Generation', 'Core', 'Display', 'Graphics', 'OS', 'Warranty'], axis=1)

# Display the cleaned data
print("\nCleaned DataFrame for Modeling:")
print(df_model.head())
print(df_model.info())
# Define features (X) and target (y)
X = df_model.drop('Price', axis=1)
y = df_model['Price']

# Identify categorical features for one-hot encoding
categorical_features = ['Brand', 'Processor Brand']

# Create a column transformer for preprocessing
# OneHotEncoder will handle categorical data, and the rest ('passthrough') will be left as is.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (Ram, SSD, Rating)
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
}

# --- Train and Evaluate Models ---
for name, model in models.items():
    # Create a pipeline that first preprocesses the data, then fits the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {name} ---")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R² Score: {r2:.4f}\n")

    # Store the best RMSE for the resume tip
    if name == "XGBoost Regressor":
        best_rmse = rmse

# --- Feature Selection for Recommendation ---
# Select numerical features that define a laptop's core specs
features_for_rec = ['Rating', 'Ram', 'SSD']
df_rec = df[features_for_rec].copy()

# --- Scaling ---
# Scale the features so that each one contributes fairly to the similarity score
scaler = StandardScaler()
df_rec_scaled = scaler.fit_transform(df_rec)

# --- Similarity Calculation ---
# Compute the cosine similarity matrix from the scaled features
cosine_sim = cosine_similarity(df_rec_scaled)

# --- Recommendation Function ---
def recommend_laptops(ram, ssd, budget, top_n=5):
    """
    Recommends top_n laptops based on user input for RAM, SSD, and Budget.
    
    Args:
        ram (int): Desired RAM in GB.
        ssd (int): Desired SSD in GB.
        budget (int): Maximum budget.
        top_n (int): Number of recommendations to return.
        
    Returns:
        pandas.DataFrame: Top N recommended laptops.
    """
    # Create a DataFrame for the user's input
    user_input = pd.DataFrame({
        'Rating': [df['Rating'].mean()], # Use average rating as a placeholder
        'Ram': [ram],
        'SSD': [ssd]
    })

    # Scale the user's input using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Calculate similarity between user input and all laptops
    sim_scores = cosine_similarity(user_input_scaled, df_rec_scaled).flatten()

    # Filter laptops within the user's budget
    budget_friendly_indices = df[df['Price'] <= budget].index
    
    # Get similarity scores for budget-friendly laptops
    filtered_sim_scores = sim_scores[budget_friendly_indices]
    
    # Get the indices of the top N most similar laptops from the filtered list
    top_indices_filtered = filtered_sim_scores.argsort()[-top_n:][::-1]
    
    # Get the original indices from the main DataFrame
    recommended_laptop_indices = budget_friendly_indices[top_indices_filtered]
    
    return df.iloc[recommended_laptop_indices]

# --- Example Usage ---
# User's desired specs
user_ram = 16  # GB
user_ssd = 512 # GB
user_budget = 70000 # Rupees

# Get recommendations
recommendations = recommend_laptops(user_ram, user_ssd, user_budget)

print("\n--- Top 5 Laptop Recommendations ---")
print(f"Based on: {user_ram}GB RAM, {user_ssd}GB SSD, and a budget of ₹{user_budget:,}")
print(recommendations[['Model', 'Price', 'Ram', 'SSD', 'Rating']])
