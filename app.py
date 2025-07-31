# Import necessary libraries
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

# --- 1. Data Loading and Cleaning ---
try:
    df = pd.read_csv('laptop.csv')
except FileNotFoundError:
    print("Error: 'laptop.csv' not found. Please place the file in the same directory as this script.")
    exit()

# Drop the 'Unnamed: 0' column
df.drop('Unnamed: 0', axis=1, inplace=True)

# Clean 'Price' column
df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(int)

# Clean 'Ram' column
df['Ram'] = df['Ram'].str.extract('(\d+)').astype(int)

# Function to clean 'SSD' column (convert TB to GB)
def convert_storage_to_gb(storage):
    if pd.isna(storage):
        return np.nan
    storage = str(storage).lower()
    if 'tb' in storage:
        num = re.findall(r'(\d+\.?\d*)', storage)
        return float(num[0]) * 1024 if num else np.nan
    elif 'gb' in storage:
        num = re.findall(r'(\d+\.?\d*)', storage)
        return float(num[0]) if num else np.nan
    return np.nan

df['SSD'] = df['SSD'].apply(convert_storage_to_gb)

# Extract 'Brand' from 'Model'
df['Brand'] = df['Model'].apply(lambda x: x.split()[0])

# Fill missing 'Rating' and 'SSD' with the median for visualization
# FIX: Replaced inplace=True to avoid FutureWarning
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['SSD'] = df['SSD'].fillna(df['SSD'].median())


# --- 2. Create Visualizations with Plotly ---

# Figure 1: Distribution of Laptop Prices
fig_price_dist = px.histogram(
    df, x='Price', nbins=50,
    title='Distribution of Laptop Prices (Right-Skewed)',
    labels={'Price': 'Price (₹)'},
    template='plotly_white'
)
fig_price_dist.update_layout(title_x=0.5)


# Figure 2: Number of Laptops by Brand
brand_counts = df['Brand'].value_counts().reset_index()
brand_counts.columns = ['Brand', 'Count']
fig_brand_count = px.bar(
    brand_counts, y='Brand', x='Count',
    orientation='h',
    title='Number of Laptops by Brand',
    labels={'Count': 'Number of Laptops', 'Brand': 'Brand'},
    template='plotly_white'
).update_yaxes(categoryorder="total ascending")
fig_brand_count.update_layout(title_x=0.5)


# Figure 3: Price Distribution by Brand
brand_order = df.groupby('Brand')['Price'].median().sort_values().index
# FIX: Replaced 'order' with 'category_orders'
fig_price_brand = px.box(
    df, x='Brand', y='Price',
    category_orders={'Brand': brand_order}, # This is the corrected argument
    title='Laptop Price Distribution by Brand',
    labels={'Price': 'Price (₹)', 'Brand': 'Brand'},
    template='plotly_white'
)
fig_price_brand.update_layout(title_x=0.5)


# Figure 4: Price vs. RAM
fig_price_ram = px.box(
    df, x='Ram', y='Price',
    title='Price vs. RAM',
    labels={'Price': 'Price (₹)', 'Ram': 'RAM (GB)'},
    template='plotly_white'
)
fig_price_ram.update_layout(title_x=0.5)


# Figure 5: Price vs. SSD
fig_price_ssd = px.scatter(
    df, x='SSD', y='Price',
    title='Price vs. SSD Storage',
    labels={'Price': 'Price (₹)', 'SSD': 'SSD (GB)'},
    template='plotly_white',
    opacity=0.6
)
fig_price_ssd.update_layout(title_x=0.5)


# Figure 6: Correlation Heatmap
numerical_df = df[['Price', 'Rating', 'Ram', 'SSD']]
corr_matrix = numerical_df.corr()
fig_corr_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Viridis',
    text=corr_matrix.round(2).values,
    texttemplate="%{text}",
    zmin=0, zmax=1
))
fig_corr_heatmap.update_layout(
    title='Correlation Matrix of Key Numerical Features',
    title_x=0.5,
    template='plotly_white'
)


# --- 3. Initialize the Dash App ---
app = Dash(__name__)
server = app.server

# --- 4. Define the Dashboard Layout ---
app.layout = html.Div(style={'backgroundColor': '#f0f2f5', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    html.Div(style={'backgroundColor': '#003366', 'padding': '20px', 'color': 'white', 'textAlign': 'center'}, children=[
        html.H1('💻 Laptop Dataset: Exploratory Data Analysis Dashboard', style={'margin': '0'})
    ]),
    
    html.Div(style={'padding': '20px'}, children=[
        
        html.Div(className='row', style={'display': 'flex', 'marginBottom': '20px'}, children=[
            dcc.Graph(id='price-distribution', figure=fig_price_dist, style={'flex': '1', 'marginRight': '10px'}),
            dcc.Graph(id='brand-count', figure=fig_brand_count, style={'flex': '1', 'marginLeft': '10px'})
        ]),

        html.Div(className='row', style={'marginBottom': '20px'}, children=[
             dcc.Graph(id='price-by-brand', figure=fig_price_brand)
        ]),

         html.Div(className='row', style={'display': 'flex', 'marginBottom': '20px'}, children=[
            dcc.Graph(id='price-vs-ram', figure=fig_price_ram, style={'flex': '1', 'marginRight': '10px'}),
            dcc.Graph(id='price-vs-ssd', figure=fig_price_ssd, style={'flex': '1', 'marginLeft': '10px'})
        ]),
        
        html.Div(className='row', style={'display': 'flex', 'justifyContent': 'center'}, children=[
            dcc.Graph(id='correlation-heatmap', figure=fig_corr_heatmap, style={'width': '60%'})
        ])
    ])
])


# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)