import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv('Yang_modelinput.csv')

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M')
df = df.sort_values('timestamp').reset_index(drop=True)

print("="*50)
print("PHASE 1: DATA PREPROCESSING")
print("="*50)

# Auto-detect feature columns
gps_cols = ['lon', 'lat']
movement_cols = [col for col in ['Distance_m', 'speed', 'direction'] if col in df.columns]
terrain_cols = [col for col in ['elevation', 'Slope', 'Aspect', 'terrainRug', 'SRTMdem'] if col in df.columns]
climate_cols = [col for col in df.columns if 'wc2.1' in col]
temp_col = [col for col in ['temperatur', 'temperature'] if col in df.columns]

# Combine all feature columns
feature_cols = gps_cols + movement_cols + terrain_cols + temp_col + climate_cols

print(f"\nDetected features:")
print(f"  GPS: {gps_cols}")
print(f"  Movement: {movement_cols}")
print(f"  Terrain: {terrain_cols}")
print(f"  Temperature: {temp_col}")
print(f"  Climate: {len(climate_cols)} variables")
print(f"\nTotal features: {len(feature_cols)}")

# Select features and handle missing values
df_features = df[feature_cols].copy()
df_features = df_features.ffill().bfill()

# Add temporal features
df_features['hour'] = df['timestamp'].dt.hour
df_features['day_of_year'] = df['timestamp'].dt.dayofyear
df_features['month'] = df['timestamp'].dt.month
df_features['timestamp'] = df['timestamp']

# Basic statistics
print(f"\nDataset Statistics:")
print(f"  Total GPS points: {len(df_features)}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"  Missing values: {df_features.isnull().sum().sum()}")

# Check data quality
if 'Distance_m' in df.columns:
    print(f"\nMovement Statistics:")
    print(f"  Average distance between points: {df['Distance_m'].mean():.2f} meters")
    print(f"  Max distance: {df['Distance_m'].max():.2f} meters")
    print(f"  Stationary points (dist < 5m): {(df['Distance_m'] < 5).sum()}")

# Save preprocessed data
df_features.to_csv('preprocessed_data.csv', index=False)
print("\n✓ Preprocessed data saved to 'preprocessed_data.csv'")
print("\nReady for Phase 2: Creating sequences")