from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
# please ignore this import
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
SEQ_LENGTH = 48
HORIZONS = [1, 6, 24]
EPOCHS = 150
BATCH_SIZE = 64
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================
# Utilities
# ============================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def haversine_distance(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance in meters between two lon/lat points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6_371_000


def create_sequences(data: np.ndarray, seq_length: int, horizons: list[int]):
    """Create input sequences and horizon targets (always lon/lat at [0:2])."""
    X, y_1h, y_6h, y_24h = [], [], [], []
    max_h = max(horizons)
    for i in range(seq_length, len(data) - max_h):
        X.append(data[i - seq_length : i])
        y_1h.append(data[i + horizons[0], 0:2])
        y_6h.append(data[i + horizons[1], 0:2])
        y_24h.append(data[i + horizons[2], 0:2])
    return np.array(X), np.array(y_1h), np.array(y_6h), np.array(y_24h)


def build_lstm_model(seq_length: int, n_features: int, name: str = "model") -> keras.Model:
    inputs = keras.Input(shape=(seq_length, n_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(96, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)

    out_1h = layers.Dense(16, activation="relu")(x)
    out_1h = layers.Dense(2, name="1h_prediction")(out_1h)
    out_6h = layers.Dense(16, activation="relu")(x)
    out_6h = layers.Dense(2, name="6h_prediction")(out_6h)
    out_24h = layers.Dense(16, activation="relu")(x)
    out_24h = layers.Dense(2, name="24h_prediction")(out_24h)

    model = keras.Model(inputs=inputs, outputs=[out_1h, out_6h, out_24h], name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={"1h_prediction": "mse", "6h_prediction": "mse", "24h_prediction": "mse"},
        loss_weights={"1h_prediction": 1.5, "6h_prediction": 1.0, "24h_prediction": 0.7},
        metrics={"1h_prediction": ["mae"], "6h_prediction": ["mae"], "24h_prediction": ["mae"]},
    )
    return model


def inverse_transform_coords(predictions: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0:2] = predictions
    return scaler.inverse_transform(dummy)[:, 0:2]


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_1h_test: np.ndarray,
    y_6h_test: np.ndarray,
    y_24h_test: np.ndarray,
    scaler: StandardScaler,
    approach_name: str,
):
    pred_1h, pred_6h, pred_24h = model.predict(X_test, verbose=0)

    pred_1h_coords = inverse_transform_coords(pred_1h, scaler)
    pred_6h_coords = inverse_transform_coords(pred_6h, scaler)
    pred_24h_coords = inverse_transform_coords(pred_24h, scaler)

    actual_1h_coords = inverse_transform_coords(y_1h_test, scaler)
    actual_6h_coords = inverse_transform_coords(y_6h_test, scaler)
    actual_24h_coords = inverse_transform_coords(y_24h_test, scaler)

    def dist_list(pred, act):
        return [haversine_distance(p[0], p[1], a[0], a[1]) for p, a in zip(pred, act)]

    errors_1h = dist_list(pred_1h_coords, actual_1h_coords)
    errors_6h = dist_list(pred_6h_coords, actual_6h_coords)
    errors_24h = dist_list(pred_24h_coords, actual_24h_coords)

    def stats(errs):
        arr = np.array(errs)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)), "rmse": float(np.sqrt((arr**2).mean())), "errors": errs}

    results = {"1h": stats(errors_1h), "6h": stats(errors_6h), "24h": stats(errors_24h)}
    log(f"Evaluated {approach_name}: 1h mean={results['1h']['mean']:.0f}m, 6h mean={results['6h']['mean']:.0f}m, 24h mean={results['24h']['mean']:.0f}m")
    return results


def accuracy_within_radius(errors: list[float], radius: float) -> float:
    arr = np.array(errors)
    return float((arr <= radius).sum() / len(arr) * 100)


# ============================
# Data
# ============================
start_time = datetime.now()
print("=" * 80)
print("ANIMAL MOVEMENT PREDICTION")
print("=" * 80)
log("Loading standardized_gps_data.csv …")

df = pd.read_csv("standardized_gps_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])  # required for hour extraction
if "hour" not in df.columns:
    df["hour"] = df["timestamp"].dt.hour

log(f"Loaded {len(df)} points; range {df['timestamp'].min()} → {df['timestamp'].max()} ({(df['timestamp'].max() - df['timestamp'].min()).days} days)")

# Feature sets
features_with_loc = [
    "lon",
    "lat",
    "Distance_m",
    "speed",
    "bearing",
    "velocity_x",
    "velocity_y",
    "elevation",
    "Slope",
    "Aspect",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_day",
    "is_weekend",
    "temperatur",
]

features_without_loc = [
    "Distance_m",
    "speed",
    "bearing",
    "velocity_x",
    "velocity_y",
    "elevation",
    "Slope",
    "Aspect",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_day",
    "is_weekend",
    "temperatur",
    "day_of_year",
    "month",
]

climate_cols = [c for c in df.columns if "wc2.1" in c][:6]
features_with_loc = [c for c in (features_with_loc + climate_cols) if c in df.columns]
features_without_loc = [c for c in (features_without_loc + climate_cols) if c in df.columns]

log(f"Features — WITH: {len(features_with_loc)}, WITHOUT: {len(features_without_loc)}")

# ============================
# Train both approaches
# ============================
all_results = {}

for approach_name, feature_list in (
    ("WITH_LOCATION", features_with_loc),
    ("WITHOUT_LOCATION", features_without_loc),
):
    print("\n" + "=" * 80)
    print(f"APPROACH: {approach_name}")
    print("=" * 80)

    df_features = df[feature_list].copy()
    if df_features.isnull().values.any():
        df_features = df_features.ffill().bfill().fillna(0)

    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(df_features)

    if approach_name == "WITHOUT_LOCATION":
        # Keep lon/lat only for target creation (not as inputs)
        lon_lat = df[["lon", "lat"]].values
        scaler_lonlat = StandardScaler()
        scaled_lonlat = scaler_lonlat.fit_transform(lon_lat)
        data_for_targets = np.concatenate([scaled_lonlat, scaled_features], axis=1)
    else:
        data_for_targets = scaled_features
        scaler_lonlat = scaler_features  # lon/lat live in first two columns

    X, y_1h, y_6h, y_24h = create_sequences(data_for_targets, SEQ_LENGTH, HORIZONS)
    if approach_name == "WITHOUT_LOCATION":
        X = X[:, :, 2:]  # drop lon/lat from inputs

    # Drop any rows with NaNs
    mask = ~(
        np.isnan(X).any(axis=(1, 2))
        | np.isnan(y_1h).any(axis=1)
        | np.isnan(y_6h).any(axis=1)
        | np.isnan(y_24h).any(axis=1)
    )
    X, y_1h, y_6h, y_24h = X[mask], y_1h[mask], y_6h[mask], y_24h[mask]

    n = len(X)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.15 * n)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y1_train, y1_val, y1_test = y_1h[:train_end], y_1h[train_end:val_end], y_1h[val_end:]
    y6_train, y6_val, y6_test = y_6h[:train_end], y_6h[train_end:val_end], y_6h[val_end:]
    y24_train, y24_val, y24_test = y_24h[:train_end], y_24h[train_end:val_end], y_24h[val_end:]

    log(f"Split — train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    model = build_lstm_model(SEQ_LENGTH, X_train.shape[2], name=approach_name)
    log(f"Params: {model.count_params():,}")

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, verbose=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=0)

    history = model.fit(
        X_train,
        {"1h_prediction": y1_train, "6h_prediction": y6_train, "24h_prediction": y24_train},
        validation_data=(X_val, {"1h_prediction": y1_val, "6h_prediction": y6_val, "24h_prediction": y24_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    log(f"Trained {len(history.history['loss'])} epochs; best val_loss={min(history.history['val_loss']):.6f}")
    model.save(f"model_{approach_name.lower()}.keras")

    results = evaluate_model(model, X_test, y1_test, y6_test, y24_test, scaler_lonlat, approach_name)
    all_results[approach_name] = {"results": results, "history": history, "model": model, "scaler": scaler_lonlat}

# ============================
# Comparison
# ============================
with_loc = all_results["WITH_LOCATION"]["results"]
without_loc = all_results["WITHOUT_LOCATION"]["results"]

print("\n" + "=" * 80)
print("COMPARISON (mean errors in meters)")
print("=" * 80)
print(f"{'Horizon':<8} {'WITH':>10} {'WITHOUT':>12} {'Winner':>10}")
for h in ["1h", "6h", "24h"]:
    w, wo = with_loc[h]["mean"], without_loc[h]["mean"]
    winner = "WITH" if w < wo else "WITHOUT"
    print(f"{h:<8} {w:>10.0f} {wo:>12.0f} {winner:>10}")

# ============================
# Pattern analysis
# ============================
log("Analyzing temporal patterns …")

hourly_stats = df.groupby("hour").agg({"Distance_m": ["mean", "count"]}).reset_index()
hourly_stats.columns = ["hour", "dist_mean", "count"]

top_hours = hourly_stats.nlargest(3, "dist_mean")
low_hours = hourly_stats.nsmallest(3, "dist_mean")

period = df["hour"].apply(lambda x: "Day" if 6 <= x <= 18 else "Night")
period_stats = df.groupby(period)["Distance_m"].mean()

# ============================
# Visualizations
# ============================
log("Creating visualizations …")

fig = plt.figure(figsize=(20, 16))

# Mean error bars
ax1 = plt.subplot(4, 3, 1)
horizons = ["1h", "6h", "24h"]
with_means = [with_loc[h]["mean"] for h in horizons]
without_means = [without_loc[h]["mean"] for h in horizons]
x = np.arange(len(horizons))
width = 0.35
ax1.bar(x - width / 2, with_means, width, label="WITH Location", alpha=0.7)
ax1.bar(x + width / 2, without_means, width, label="WITHOUT Location", alpha=0.7)
ax1.set_ylabel("Mean Error (m)")
ax1.set_title("Prediction Error Comparison")
ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")
for bars in ax1.containers:
    ax1.bar_label(bars, fmt="%.0fm", padding=2, fontsize=8)

# Val loss curves
ax2 = plt.subplot(4, 3, 2)
ax2.plot(all_results["WITH_LOCATION"]["history"].history["val_loss"], label="WITH Location")
ax2.plot(all_results["WITHOUT_LOCATION"]["history"].history["val_loss"], label="WITHOUT Location")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Loss")
ax2.set_title("Training Convergence")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Accuracy vs radius (1h)
ax3 = plt.subplot(4, 3, 3)
radii = [500, 1000, 2000, 5000]
with_acc = [accuracy_within_radius(with_loc["1h"]["errors"], r) for r in radii]
without_acc = [accuracy_within_radius(without_loc["1h"]["errors"], r) for r in radii]
ax3.plot(radii, with_acc, "o-", label="WITH Location")
ax3.plot(radii, without_acc, "s-", label="WITHOUT Location")
ax3.set_xlabel("Radius (m)")
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Accuracy vs Distance Threshold (1h)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Error histograms WITH
for idx, (h, color) in enumerate([("1h", "tab:blue"), ("6h", "tab:orange"), ("24h", "tab:red")]):
    ax = plt.subplot(4, 3, 4 + idx)
    errs = with_loc[h]["errors"]
    ax.hist(errs, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(errs), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(errs):.0f}m")
    ax.axvline(np.median(errs), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(errs):.0f}m")
    ax.set_title(f"WITH Location — {h} Errors")
    ax.set_xlabel("Distance Error (m)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Error histograms WITHOUT
for idx, (h, color) in enumerate([("1h", "tab:blue"), ("6h", "tab:orange"), ("24h", "tab:red")]):
    ax = plt.subplot(4, 3, 7 + idx)
    errs = without_loc[h]["errors"]
    ax.hist(errs, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(errs), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(errs):.0f}m")
    ax.axvline(np.median(errs), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(errs):.0f}m")
    ax.set_title(f"WITHOUT Location — {h} Errors")
    ax.set_xlabel("Distance Error (m)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hourly movement
ax10 = plt.subplot(4, 3, 10)
ax10.bar(hourly_stats["hour"], hourly_stats["dist_mean"], alpha=0.7)
ax10.set_xlabel("Hour")
ax10.set_ylabel("Avg Distance (m)")
ax10.set_title("Hourly Movement Pattern")
ax10.grid(True, alpha=0.3, axis="y")
ax10.set_xticks(range(0, 24, 3))

# Temp vs movement (if available)
ax11 = plt.subplot(4, 3, 11)
if "temperatur" in df.columns:
    sc = ax11.scatter(df["temperatur"], df["Distance_m"], alpha=0.3, s=5, c=df["hour"], cmap="viridis")
    ax11.set_xlabel("Temperature (°C)")
    ax11.set_ylabel("Distance (m)")
    ax11.set_title("Temperature vs Movement")

# Behavioral states
ax12 = plt.subplot(4, 3, 12)
states = pd.cut(df["Distance_m"], bins=[0, 50, 200, 1000, 10000], labels=["Stationary", "Resting", "Active", "Traveling"])
state_counts = states.value_counts()
ax12.pie(state_counts.values, labels=state_counts.index, autopct="%1.1f%%", startangle=90)
ax12.set_title("Behavioral States")

plt.tight_layout()
plt.savefig("final_comprehensive_results.png", dpi=300, bbox_inches="tight")
log("Saved final_comprehensive_results.png")

# ============================
# Report
# ============================
log("Writing report …")

best = "WITH_LOCATION" if with_loc["1h"]["mean"] < without_loc["1h"]["mean"] else "WITHOUT_LOCATION"
best_results = all_results[best]["results"]

dur_days = (df["timestamp"].max() - df["timestamp"].min()).days
report = f"""
COMPREHENSIVE ANIMAL MOVEMENT PREDICTION — FINAL REPORT
Submitted by: Ravindra Nath Tripathi
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analyzed: {dur_days} days of GPS data

Executive Summary
— Two LSTM approaches compared: WITH location vs WITHOUT location.
— Best (1‑hour horizon): {best}
— 1h/6h/24h mean errors: {best_results['1h']['mean']:.0f}m / {best_results['6h']['mean']:.0f}m / {best_results['24h']['mean']:.0f}m

Detailed Comparison (mean meters)
  1h: WITH {with_loc['1h']['mean']:.0f}  | WITHOUT {without_loc['1h']['mean']:.0f}
  6h: WITH {with_loc['6h']['mean']:.0f}  | WITHOUT {without_loc['6h']['mean']:.0f}
 24h: WITH {with_loc['24h']['mean']:.0f}  | WITHOUT {without_loc['24h']['mean']:.0f}
Within 1km (1h): WITH {accuracy_within_radius(with_loc['1h']['errors'], 1000):.1f}% | WITHOUT {accuracy_within_radius(without_loc['1h']['errors'], 1000):.1f}%

Patterns
— Most active hours: {', '.join([f"{int(h)}:00 ({d:.0f}m)" for h, d in zip(top_hours['hour'], top_hours['dist_mean'])])}
— Least active hours: {', '.join([f"{int(h)}:00 ({d:.0f}m)" for h, d in zip(low_hours['hour'], low_hours['dist_mean'])])}
— Day vs Night (avg m/h): Day {period_stats.get('Day', float('nan')):.0f} | Night {period_stats.get('Night', float('nan')):.0f}

Practical Takeaways
— WITH location: best short‑term tracking and gap‑filling.
— WITHOUT location: better for understanding environmental drivers.
— Use both: prediction + interpretation.

Artifacts
— model_with_location.keras
— model_without_location.keras
— final_comprehensive_results.png
— comprehensive_final_report.txt
"""

with open("comprehensive_final_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
log("Saved comprehensive_final_report.txt")

# ============================
# Summary
# ============================
end_time = datetime.now()
elapsed_min = (end_time - start_time).total_seconds() / 60

print("\n" + "=" * 80)
print("DONE ✅")
print("=" * 80)
print(
    f"Best approach: {best}\n"
    f"1h: {best_results['1h']['mean']:.0f}m | 6h: {best_results['6h']['mean']:.0f}m | 24h: {best_results['24h']['mean']:.0f}m\n"
    f"Elapsed: {elapsed_min:.1f} min\n"
    f"Files: model_with_location.keras, model_without_location.keras, final_comprehensive_results.png, comprehensive_final_report.txt\n"
)
