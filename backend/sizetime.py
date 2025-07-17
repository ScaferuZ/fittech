import pickle
import numpy as np
import time
import psutil
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Try to import psutil for memory measurement
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    print("[WARNING] psutil not installed. Memory usage measurement will be skipped.")

# Paths to your model files
xgb_path = "models/xgfitness_ai_model.pkl"
rf_path = "models/research_model_comparison.pkl"

# Load XGBoost model only
with open(xgb_path, "rb") as f:
    xgb_data = pickle.load(f)
    xgb_model = xgb_data["workout_model"]
    xgb_scaler = xgb_data["scaler"]
    xgb_features = xgb_data["feature_columns"]

# Load Random Forest model only (ignore XGBoost inside)
with open(rf_path, "rb") as f:
    rf_data = pickle.load(f)
    rf_model = rf_data["workout_rf_model"]
    rf_scaler = rf_data["rf_scaler"]
    rf_features = rf_data["feature_columns"]

# Dummy test data (replace with your real test set for accuracy)
X_test = np.random.rand(1000, len(xgb_features))

# Scale data
X_test_xgb = xgb_scaler.transform(X_test)
X_test_rf = rf_scaler.transform(X_test)

def measure_inference(model, X):
    start = time.time()
    model.predict(X)
    end = time.time()
    return (end - start) / len(X) * 1000  # ms per sample

def measure_memory(model, X):
    if not psutil_available:
        return None
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    model.predict(X)
    mem_after = process.memory_info().rss / 1024**2
    return mem_after - mem_before

# Model file sizes
xgb_size = os.path.getsize(xgb_path) / 1024**2
rf_size = os.path.getsize(rf_path) / 1024**2
rf_only_size = rf_size - xgb_size

# Inference speed
xgb_time = measure_inference(xgb_model, X_test_xgb)
rf_time = measure_inference(rf_model, X_test_rf)

# Memory usage (if psutil available)
xgb_mem = measure_memory(xgb_model, X_test_xgb)
rf_mem = measure_memory(rf_model, X_test_rf)

# Training time benchmarking (using dummy data)
X_train = np.random.rand(2000, len(xgb_features))
y_train = np.random.randint(0, 9, 2000)  # 9 classes for workout template

# XGBoost training time
xgb_train_model = XGBClassifier(tree_method='hist', n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
start = time.time()
xgb_train_model.fit(X_train, y_train)
xgb_train_time = time.time() - start

# Random Forest training time
rf_train_model = RandomForestClassifier(n_jobs=-1)
start = time.time()
rf_train_model.fit(X_train, y_train)
rf_train_time = time.time() - start

print("\nModel Efficiency Benchmark Table:")
print("| Model                | File Size (MB) | Inference Time (ms/sample) | Memory Usage (MB) | Training Time (s) |")
print("|----------------------|:--------------:|:-------------------------:|:-----------------:|:-----------------:|")
print(f"| XGBoost (Primary)    | {xgb_size:.2f}           | {xgb_time:.4f}                  | {xgb_mem if xgb_mem is not None else 'N/A'}           | {xgb_train_time:.2f}           |")
print(f"| Random Forest (Full) | {rf_size:.2f}           | {rf_time:.4f}                  | {rf_mem if rf_mem is not None else 'N/A'}           | {rf_train_time:.2f}           |")
print(f"| Random Forest (Only) | {rf_only_size:.2f}           | (same as above)                | (same as above)           | (same as above)           |")
print("\nNotes:")
print("- 'Random Forest (Only)' size is estimated as (RF full size - XGBoost size)")
print("- Inference time, memory usage, and training time are measured on dummy data. Use real data for publication.")
if not psutil_available:
    print("- Memory usage not measured (psutil not installed)")