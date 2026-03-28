# %%
# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import warnings
import joblib

from bayes_opt import BayesianOptimization
from sklearn.neighbors import KNeighborsRegressor

# ======================
# Scikit-learn
# ======================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# ======================
# Settings
# ======================
warnings.simplefilter(action="ignore")

# ======================
# 1. Global settings
# ======================
DATA_PATH = "data_1218_scaling.csv"
MODEL_PATH = "knn_final_model.pkl"
FOLD_RESULTS_PATH = "knn_fold_results.csv"
TEST_RESULT_PATH = "knn_oof_predictions.csv"
FEATURE_START_COL = 8  # Features start from the 7th column (0-indexed)

RANDOM_STATE = 42
N_SPLITS = 10

# ======================
# 2. Utility: evaluation
# ======================
def evaluate_performance(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# ======================
# 3. Load data
# ======================
data = pd.read_csv(DATA_PATH)

X = data.iloc[:, FEATURE_START_COL:]
y = data["value"]

# ======================
# 4. CV preparation
# ======================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

train_metrics_list = []
val_metrics_list = []
test_metrics_list = []
fold_details_list = []

# OOF predictions
all_test_predictions_full = np.zeros(len(data))

# ======================
# 5. Cross-validation
# ======================
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'='*25} Fold {fold + 1} {'='*25}")

    # ---- split data ----
    remain_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    X_remain = remain_data.iloc[:, FEATURE_START_COL:]
    y_remain = remain_data["value"]

    X_test = test_data.iloc[:, FEATURE_START_COL:]
    y_test = test_data["value"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_remain,
        y_remain,
        test_size=0.1,
        random_state=RANDOM_STATE
    )

    # ======================
    # 6. Bayesian objective
    # ======================
    def knn_objective(n_neighbors, weights_flag, p):
        weights = "uniform" if weights_flag < 0.5 else "distance"

        model = KNeighborsRegressor(
            n_neighbors=int(n_neighbors),
            weights=weights,
            p=int(round(p))
        )

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_val_pred)

        # BO maximization → negative MAE
        return -score

    # ======================
    # 7. Bayesian Optimization
    # ======================
    pbounds = {
        "n_neighbors": (10, 100),
        "weights_flag": (0.0, 1.0),
        "p": (1, 2),
    }

    optimizer = BayesianOptimization(
        f=knn_objective,
        pbounds=pbounds,
        random_state=RANDOM_STATE
    )

    optimizer.maximize(init_points=20, n_iter=200)

    best_params = optimizer.max["params"]
    best_params["n_neighbors"] = int(best_params["n_neighbors"])
    best_params["p"] = int(round(best_params["p"]))
    best_params["weights"] = (
        "uniform" if best_params["weights_flag"] < 0.5 else "distance"
    )

    print(f"Best CV MAE: {-optimizer.max['target']:.4f}")
    print("Best params:", best_params)

    # ======================
    # 8. Train best model
    # ======================
    best_model = KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        p=best_params["p"]
    )

    # ---- validation ----
    best_model.fit(X_train, y_train)
    val_metrics = evaluate_performance(best_model, X_val, y_val)

    # ---- retrain on remain ----
    best_model.fit(X_remain, y_remain)

    train_metrics = evaluate_performance(best_model, X_remain, y_remain)
    test_metrics = evaluate_performance(best_model, X_test, y_test)

    # ======================
    # 9. Save OOF predictions
    # ======================
    y_test_pred = best_model.predict(X_test)
    all_test_predictions_full[test_idx] = y_test_pred

    # ======================
    # 10. Save metrics
    # ======================
    train_metrics_list.append(train_metrics)
    val_metrics_list.append(val_metrics)
    test_metrics_list.append(test_metrics)

    fold_details_list.append({
        "fold": fold + 1,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    })

    print("Train:", train_metrics)
    print("Val:  ", val_metrics)
    print("Test: ", test_metrics)

# ======================
# 11. Save fold results
# ======================
fold_details_df = pd.DataFrame(fold_details_list)
fold_details_df.to_csv(FOLD_RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ Fold details saved to {FOLD_RESULTS_PATH}")

# ======================
# 12. Save OOF predictions
# ======================
data_with_pred = data.copy()
data_with_pred.insert(0, "predicted_value", all_test_predictions_full)
data_with_pred.to_csv(TEST_RESULT_PATH, index=False, encoding="utf-8-sig")
print("✅ OOF test predictions saved")

# ======================
# 13. Summary statistics
# ======================
print("\n" + "=" * 80)
print("Cross-Validation Performance Summary")
print("=" * 80)

train_df = pd.DataFrame(train_metrics_list)
val_df = pd.DataFrame(val_metrics_list)
test_df = pd.DataFrame(test_metrics_list)

print("\n=== Train ===")
print(train_df.mean(), "\n±\n", train_df.std())

print("\n=== Validation ===")
print(val_df.mean(), "\n±\n", val_df.std())

print("\n=== Test ===")
print(test_df.mean(), "\n±\n", test_df.std())

# ======================
# 14. Train & save final model
# ======================
print("\nTraining final KNN model on full dataset...")

final_model = KNeighborsRegressor(
    n_neighbors=best_params["n_neighbors"],
    weights=best_params["weights"],
    p=best_params["p"]
)

final_model.fit(X, y)
joblib.dump(final_model, MODEL_PATH)

print(f"✅ Final model saved to {MODEL_PATH}")



