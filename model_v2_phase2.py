import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

print("="*50)
print("PHASE 2: SEQUENCE CREATION")
print("="*50)

# Load preprocessed data
df_features = pd.read_csv('preprocessed_data.csv')
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

print(f"\nLoaded {len(df_features)} data points")

# Remove timestamp for now
timestamps = df_features['timestamp'].values
df_features_no_time = df_features.drop('timestamp', axis=1)

print(f"Features: {len(df_features_no_time.columns)}")

# ============================================
# CRITICAL FIX: Clean the data
# ============================================

print("\n✓ Cleaning data...")

# Replace Excel errors and invalid values
df_features_no_time = df_features_no_time.replace(['#DIV/0!', '#VALUE!', '#N/A', '#REF!', '#NUM!'], np.nan)

# Convert all columns to numeric (coerce errors to NaN)
for col in df_features_no_time.columns:
    df_features_no_time[col] = pd.to_numeric(df_features_no_time[col], errors='coerce')

# Check for NaN values
nan_counts = df_features_no_time.isnull().sum()
print(f"\nNaN values per column:")
for col in nan_counts[nan_counts > 0].index:
    print(f"  {col}: {nan_counts[col]}")

# Fill NaN values with forward fill, then backward fill, then median
df_features_no_time = df_features_no_time.ffill().bfill()

# If still NaN (e.g., entire column), fill with median or 0
for col in df_features_no_time.columns:
    if df_features_no_time[col].isnull().any():
        median_val = df_features_no_time[col].median()
        if pd.isna(median_val):
            df_features_no_time[col].fillna(0, inplace=True)
        else:
            df_features_no_time[col].fillna(median_val, inplace=True)

# Check for infinite values
df_features_no_time = df_features_no_time.replace([np.inf, -np.inf], np.nan)
df_features_no_time = df_features_no_time.fillna(0)

print(f"✓ Data cleaned. Remaining NaN: {df_features_no_time.isnull().sum().sum()}")

# Verify all values are numeric
print(f"✓ Data types check:")
print(df_features_no_time.dtypes.value_counts())

# Normalize all features
print("\n✓ Normalizing features...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features_no_time)

print(f"✓ Data normalized. Shape: {scaled_data.shape}")

# ============================================
# Create sequences function
# ============================================

def create_sequences(data, seq_length=48, horizons=[1, 6, 24]):
    """
    Create sequences for multi-horizon prediction
    seq_length: number of past points to use
    horizons: future steps to predict [1, 6, 24]
    """
    X, y_1h, y_6h, y_24h = [], [], [], []
    
    max_horizon = max(horizons)
    
    for i in range(seq_length, len(data) - max_horizon):
        # Input: past seq_length points
        X.append(data[i-seq_length:i])
        
        # Targets: lat/lon at future horizons (first 2 columns are lon, lat)
        y_1h.append(data[i + horizons[0], 0:2])
        y_6h.append(data[i + horizons[1], 0:2])
        y_24h.append(data[i + horizons[2], 0:2])
    
    return np.array(X), np.array(y_1h), np.array(y_6h), np.array(y_24h)

# Create sequences
SEQ_LENGTH = 48  # Use past 48 points
HORIZONS = [1, 6, 24]  # 1, 6, 24 steps ahead

print(f"\nCreating sequences...")
print(f"  Sequence length: {SEQ_LENGTH} past points")
print(f"  Prediction horizons: {HORIZONS} steps ahead")

X, y_1h, y_6h, y_24h = create_sequences(scaled_data, seq_length=SEQ_LENGTH, horizons=HORIZONS)

print(f"\n✓ Sequences created:")
print(f"  X shape: {X.shape}")
print(f"  y_1h shape: {y_1h.shape}")
print(f"  y_6h shape: {y_6h.shape}")
print(f"  y_24h shape: {y_24h.shape}")

# Verify no NaN in sequences
if np.isnan(X).any() or np.isnan(y_1h).any():
    print("\n⚠ WARNING: NaN detected in sequences. Removing affected samples...")
    # Find valid samples (no NaN)
    valid_mask = ~(np.isnan(X).any(axis=(1,2)) | np.isnan(y_1h).any(axis=1) | 
                   np.isnan(y_6h).any(axis=1) | np.isnan(y_24h).any(axis=1))
    X = X[valid_mask]
    y_1h, y_6h, y_24h = y_1h[valid_mask], y_6h[valid_mask], y_24h[valid_mask]
    print(f"  Cleaned sequences: {X.shape[0]} samples")

# Train/Validation/Test split (temporal split)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_1h_train, y_6h_train, y_24h_train = y_1h[:train_size], y_6h[:train_size], y_24h[:train_size]

X_val = X[train_size:train_size+val_size]
y_1h_val, y_6h_val, y_24h_val = y_1h[train_size:train_size+val_size], y_6h[train_size:train_size+val_size], y_24h[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_1h_test, y_6h_test, y_24h_test = y_1h[train_size+val_size:], y_6h[train_size+val_size:], y_24h[train_size+val_size:]

print(f"\n✓ Data split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Testing: {len(X_test)} samples")

# ============================================
# PHASE 3: BUILD MODEL
# ============================================

print("\n" + "="*50)
print("PHASE 3: MODEL BUILDING")
print("="*50)

def build_multi_horizon_lstm(seq_length, n_features):
    """
    Multi-output LSTM model for 1h, 6h, 24h predictions
    """
    inputs = keras.Input(shape=(seq_length, n_features), name='input')
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=True, name='lstm_2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32, name='lstm_3')(x)
    x = layers.Dropout(0.2)(x)
    
    # Shared dense layer
    x = layers.Dense(64, activation='relu', name='shared_dense')(x)
    x = layers.Dropout(0.2)(x)
    
    # Three separate output heads
    output_1h = layers.Dense(32, activation='relu')(x)
    output_1h = layers.Dense(2, name='1h_prediction')(output_1h)
    
    output_6h = layers.Dense(32, activation='relu')(x)
    output_6h = layers.Dense(2, name='6h_prediction')(output_6h)
    
    output_24h = layers.Dense(32, activation='relu')(x)
    output_24h = layers.Dense(2, name='24h_prediction')(output_24h)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[output_1h, output_6h, output_24h],
        name='AnimalTrajectory_MultiHorizon'
    )
    
    return model

# Build model
n_features = X_train.shape[2]
model = build_multi_horizon_lstm(SEQ_LENGTH, n_features)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        '1h_prediction': 'mse',
        '6h_prediction': 'mse',
        '24h_prediction': 'mse'
    },
    loss_weights={
        '1h_prediction': 1.0,
        '6h_prediction': 0.8,
        '24h_prediction': 0.6
    },
    metrics={
        '1h_prediction': ['mae'],
        '6h_prediction': ['mae'],
        '24h_prediction': ['mae']
    }
)

print("\n✓ Model architecture:")
model.summary()

# ============================================
# PHASE 4: TRAIN MODEL
# ============================================

print("\n" + "="*50)
print("PHASE 4: MODEL TRAINING")
print("="*50)

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train model
print("\nStarting training...")
print("This may take 30-60 minutes depending on your hardware...")

history = model.fit(
    X_train,
    {
        '1h_prediction': y_1h_train,
        '6h_prediction': y_6h_train,
        '24h_prediction': y_24h_train
    },
    validation_data=(
        X_val,
        {
            '1h_prediction': y_1h_val,
            '6h_prediction': y_6h_val,
            '24h_prediction': y_24h_val
        }
    ),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

print("\n✓ Training completed!")
print(f"  Total epochs: {len(history.history['loss'])}")
print(f"  Best validation loss: {min(history.history['val_loss']):.6f}")

# Save final model
model.save('animal_trajectory_final_model.keras')
print("\n✓ Model saved!")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['1h_prediction_loss'], label='Train')
plt.plot(history.history['val_1h_prediction_loss'], label='Val')
plt.title('1-Hour Prediction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['6h_prediction_loss'], label='Train')
plt.plot(history.history['val_6h_prediction_loss'], label='Val')
plt.title('6-Hour Prediction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['24h_prediction_loss'], label='Train')
plt.plot(history.history['val_24h_prediction_loss'], label='Val')
plt.title('24-Hour Prediction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\n✓ Training plots saved as 'training_history.png'")

# ============================================
# PHASE 5: EVALUATION
# ============================================

print("\n" + "="*50)
print("PHASE 5: MODEL EVALUATION")
print("="*50)

# Make predictions
print("\nGenerating predictions on test set...")
pred_1h, pred_6h, pred_24h = model.predict(X_test, verbose=0)

# Inverse transform to get actual coordinates
def inverse_transform_coords(predictions, scaler_obj):
    """Convert normalized predictions back to lat/lon"""
    dummy = np.zeros((len(predictions), scaler_obj.n_features_in_))
    dummy[:, 0:2] = predictions
    inversed = scaler_obj.inverse_transform(dummy)
    return inversed[:, 0:2]

pred_1h_coords = inverse_transform_coords(pred_1h, scaler)
pred_6h_coords = inverse_transform_coords(pred_6h, scaler)
pred_24h_coords = inverse_transform_coords(pred_24h, scaler)

actual_1h_coords = inverse_transform_coords(y_1h_test, scaler)
actual_6h_coords = inverse_transform_coords(y_6h_test, scaler)
actual_24h_coords = inverse_transform_coords(y_24h_test, scaler)

# Haversine distance calculation
def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two GPS points"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

# Calculate errors
print("\nCalculating prediction errors...")
errors_1h = [haversine(p[0], p[1], a[0], a[1]) 
             for p, a in zip(pred_1h_coords, actual_1h_coords)]
errors_6h = [haversine(p[0], p[1], a[0], a[1]) 
             for p, a in zip(pred_6h_coords, actual_6h_coords)]
errors_24h = [haversine(p[0], p[1], a[0], a[1]) 
              for p, a in zip(pred_24h_coords, actual_24h_coords)]

# Results
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

print(f"\n1-HOUR PREDICTION:")
print(f"  Mean Error: {np.mean(errors_1h):.2f} meters")
print(f"  Median Error: {np.median(errors_1h):.2f} meters")
print(f"  RMSE: {np.sqrt(np.mean(np.array(errors_1h)**2)):.2f} meters")

print(f"\n6-HOUR PREDICTION:")
print(f"  Mean Error: {np.mean(errors_6h):.2f} meters")
print(f"  Median Error: {np.median(errors_6h):.2f} meters")
print(f"  RMSE: {np.sqrt(np.mean(np.array(errors_6h)**2)):.2f} meters")

print(f"\n24-HOUR PREDICTION:")
print(f"  Mean Error: {np.mean(errors_24h):.2f} meters")
print(f"  Median Error: {np.median(errors_24h):.2f} meters")
print(f"  RMSE: {np.sqrt(np.mean(np.array(errors_24h)**2)):.2f} meters")

# Accuracy within radius
def accuracy_within_radius(errors, radius):
    return (np.array(errors) <= radius).sum() / len(errors) * 100

print(f"\nACCURACY WITHIN RADIUS:")
for radius in [50, 100, 250, 500, 1000]:
    acc_1h = accuracy_within_radius(errors_1h, radius)
    acc_6h = accuracy_within_radius(errors_6h, radius)
    acc_24h = accuracy_within_radius(errors_24h, radius)
    print(f"  Within {radius}m: 1h={acc_1h:.1f}%, 6h={acc_6h:.1f}%, 24h={acc_24h:.1f}%")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(errors_1h, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(errors_1h), color='red', linestyle='--', label=f'Mean: {np.mean(errors_1h):.1f}m')
axes[0].set_title('1-Hour Prediction Error', fontweight='bold')
axes[0].set_xlabel('Distance Error (meters)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(errors_6h, bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(np.mean(errors_6h), color='red', linestyle='--', label=f'Mean: {np.mean(errors_6h):.1f}m')
axes[1].set_title('6-Hour Prediction Error', fontweight='bold')
axes[1].set_xlabel('Distance Error (meters)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(errors_24h, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[2].axvline(np.mean(errors_24h), color='red', linestyle='--', label=f'Mean: {np.mean(errors_24h):.1f}m')
axes[2].set_title('24-Hour Prediction Error', fontweight='bold')
axes[2].set_xlabel('Distance Error (meters)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved!")

# Save results
results_text = f"""
ANIMAL MOVEMENT PREDICTION - RESULTS
=====================================

1-Hour: Mean={np.mean(errors_1h):.2f}m, Median={np.median(errors_1h):.2f}m
6-Hour: Mean={np.mean(errors_6h):.2f}m, Median={np.median(errors_6h):.2f}m
24-Hour: Mean={np.mean(errors_24h):.2f}m, Median={np.median(errors_24h):.2f}m

Model: Multi-horizon LSTM with {n_features} features
Training samples: {len(X_train)}
Test samples: {len(X_test)}
"""

with open('results_summary.txt', 'w') as f:
    f.write(results_text)

print("\n" + "="*50)
print("✓✓✓ ALL DONE! ✓✓✓")
print("="*50)