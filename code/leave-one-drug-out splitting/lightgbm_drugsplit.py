# %%
# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import GroupKFold


from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor

# ======================
# Scikit-learn
# ======================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    train_test_split,
    ShuffleSplit,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

# ======================
# Settings
# ======================
warnings.simplefilter(action="ignore")


# ======================
# 1. Global settings
# ======================
DATA_PATH = "data_1218_processing.csv"
MODEL_PATH = "lgbm_final_model.pkl"
FOLD_RESULTS_PATH = "lgbm_fold_results.csv"
TEST_RESULT_PATH = "lgbm_oof_predictions.csv"
FEATURE_START_COL = 8  # Features start from the 7th column (0-indexed)

RANDOM_STATE = 42
N_SPLITS = 65

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

groups = data['GROUP']
group_kfold = GroupKFold(n_splits=N_SPLITS,shuffle=True, random_state=RANDOM_STATE)

# ======================

# ======================
# 4. CV preparation
# ======================

train_metrics_list = []
val_metrics_list = []
test_metrics_list = []
fold_details_list = []

train_predictions_dict = {}
val_predictions_dict = {}
test_predictions_dict = {}

# OOF test predictions (TabPFN-style)
all_test_predictions_full = np.zeros(len(data))

# ======================
# 5. Cross-validation
# ======================
for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
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
    def lgb_objective(
        n_estimators,
        learning_rate,
        subsample,
        colsample_bytree,
        reg_alpha,
        reg_lambda,
        num_leaves,
        subsample_freq,
        colsample_bynode
    ):
        model = LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            subsample=subsample,
            subsample_freq=int(subsample_freq),
            num_leaves=int(num_leaves),
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE, verbose = -1
        )

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)

        return - score

    # ======================
    # 7. Bayesian Optimization
    # ======================
    pbounds = {
        "n_estimators": (100, 1000),
        "learning_rate": (0.0005, 0.1),
        "num_leaves": (2, 50),
        "subsample": (0.5, 1.0),
        "subsample_freq": (1, 5),
        "colsample_bytree": (0.1, 1.0),
        "colsample_bynode": (0.1, 1.0),
        "reg_alpha": (0.0, 5.0),
        "reg_lambda": (0.0, 10.0),
    }

    optimizer = BayesianOptimization(
        f=lgb_objective,
        pbounds=pbounds,
        random_state=RANDOM_STATE
    )

    optimizer.maximize(init_points=20, n_iter=200)

    best_params = optimizer.max["params"]
    print(f"Best CV score: {optimizer.max['target']:.4f}")

    # ======================
    # 8. Train best model
    # ======================
    best_model = LGBMRegressor(
        n_estimators=int(best_params["n_estimators"]),
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        subsample_freq=int(best_params["subsample_freq"]),
        num_leaves=int(best_params["num_leaves"]),
        colsample_bytree=best_params["colsample_bytree"],
        colsample_bynode=best_params["colsample_bynode"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        random_state=RANDOM_STATE,  verbose = -1  

    )

    # ---- validation ----
    best_model.fit(X_train, y_train)
    val_metrics = evaluate_performance(best_model, X_val, y_val)

    # ---- retrain on remain ----
    best_model.fit(X_remain, y_remain)

    train_metrics = evaluate_performance(best_model, X_remain, y_remain)
    test_metrics = evaluate_performance(best_model, X_test, y_test)

    # ======================
    # 9. Save predictions
    # ======================
    y_train_pred = best_model.predict(X_remain)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    train_predictions_dict[f"fold_{fold+1}"] = {
        "indices": train_idx,
        "y_true": y_remain.values,
        "y_pred": y_train_pred
    }

    val_predictions_dict[f"fold_{fold+1}"] = {
        "indices": X_val.index.values,
        "y_true": y_val.values,
        "y_pred": y_val_pred
    }

    test_predictions_dict[f"fold_{fold+1}"] = {
        "indices": test_idx,
        "y_true": y_test.values,
        "y_pred": y_test_pred
    }

    # OOF prediction
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
print("\nTraining final LightGBM model on full dataset...")

final_model = LGBMRegressor(
    n_estimators=int(best_params["n_estimators"]),
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    subsample_freq=int(best_params["subsample_freq"]),
    num_leaves=int(best_params["num_leaves"]),
    colsample_bytree=best_params["colsample_bytree"],
    colsample_bynode=best_params["colsample_bynode"],
    reg_alpha=best_params["reg_alpha"],
    reg_lambda=best_params["reg_lambda"],
    random_state=RANDOM_STATE
)

final_model.fit(X, y)
joblib.dump(final_model, MODEL_PATH)

print(f"✅ Final model saved to {MODEL_PATH}")


