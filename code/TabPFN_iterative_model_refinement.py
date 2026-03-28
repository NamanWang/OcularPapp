# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.model_selection import GroupKFold
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy import stats

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabpfn import TabPFNRegressor

# =========================
# Parameter settings
# =========================
N_ROUNDS = 40          # i = 1 ~ 10
N_REPEAT = 10          # 10 random repetitions per round
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================
# Utility functions
# =========================
def load_xy(csv_path):
    """
    Default: the first column is y, the rest are X
    """
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 7].values
    X = df.iloc[:, 8:].values
    return X, y, df


def evaluate_model(X_train, y_train, X_test, y_test):
    model = TabPFNRegressor(device="cuda",ignore_pretraining_limits=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)

    return r2, mae, rmse


# =========================
# Result storage
# =========================
all_results = []

# =========================
# Round 0
# =========================
print("Running round 0 ...")

X_train, y_train, train_df = load_xy("train_0.csv")
X_test, y_test, test_df = load_xy("test_0.csv")

r2, mae, rmse = evaluate_model(X_train, y_train, X_test, y_test)

all_results.append({
    "round": 0,
    "repeat": 0,
    "R2": r2,
    "MAE": mae,
    "RMSE": rmse
})

# =========================
# Round i (i = 1 ~ 10)
# =========================
for i in range(1, N_ROUNDS + 1):
    print(f"\nRunning round {i} ...")

    prev_train = pd.read_csv(f"train_0.csv")
    prev_test = pd.read_csv(f"test_0.csv")

    metrics_repeat = []

    for rep in range(N_REPEAT):
        # ---- Random sampling ----
        sampled_idx = np.random.choice(
            prev_test.index, size=i, replace=False
        )

        add_df = prev_test.loc[sampled_idx]
        new_test_df = prev_test.drop(sampled_idx)
        new_train_df = pd.concat([prev_train, add_df], axis=0).reset_index(drop=True)

        # ---- Save intermediate files (can be removed if not needed) ----
        train_i_path = f"train_{i}.csv"
        test_i_path = f"test_{i}.csv"
        new_train_df.to_csv(train_i_path, index=False)
        new_test_df.to_csv(test_i_path, index=False)

        # ---- Training & testing ----
        X_train, y_train, _ = load_xy(train_i_path)
        X_test, y_test, _ = load_xy(test_i_path)

        r2, mae, rmse = evaluate_model(X_train, y_train, X_test, y_test)

        metrics_repeat.append((r2, mae, rmse))

        all_results.append({
            "round": i,
            "repeat": rep + 1,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })

    # ---- Calculate mean & standard deviation ----
    metrics_repeat = np.array(metrics_repeat)

    all_results.append({
        "round": i,
        "repeat": "mean",
        "R2": metrics_repeat[:, 0].mean(),
        "MAE": metrics_repeat[:, 1].mean(),
        "RMSE": metrics_repeat[:, 2].mean()
    })

    all_results.append({
        "round": i,
        "repeat": "std",
        "R2": metrics_repeat[:, 0].std(),
        "MAE": metrics_repeat[:, 1].std(),
        "RMSE": metrics_repeat[:, 2].std()
    })

# =========================
# Save final results
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv("performance_result.csv", index=False)

print("\nAll experiments finished.")
print("Results saved to performance_result.csv")