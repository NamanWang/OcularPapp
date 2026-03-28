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

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = 'data_1218_processing.csv'
MODEL_PATH = "TabPFN_model.pkl"
RESULTS_PATH = "TabPFN_fold_results.csv"
RANDOM_STATE = 42
DEVICE = 'cuda'
TEST_RESULT = "TabPFN_oof_predictions.csv"
FEATURE_START_COL = 8  # Features start from the 7th column (0-indexed)

data = pd.read_csv(DATA_PATH)

X = data.iloc[:, FEATURE_START_COL:]
print(X.columns)
y = data['value']
groups = data['NUMBER']

group_kfold = GroupKFold(n_splits=10)

train_mse_list, train_mae_list, train_r2_list, train_rmse_list = [], [], [], []
test_mse_list, test_mae_list, test_r2_list, test_rmse_list = [], [], [], []

train_predictions_dict, test_predictions_dict = {}, {}
fold_details_list = []
all_test_predictions_full = np.zeros(len(data))


for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    reg = TabPFNRegressor(device=DEVICE)
    reg.fit(X_train, y_train)

    train_predictions = reg.predict(X_train)
    test_predictions = reg.predict(X_test)
    all_test_predictions_full[test_idx] = test_predictions

    train_mse = mean_squared_error(y_train, train_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)

    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)

    train_mse_list.append(train_mse)
    train_mae_list.append(train_mae)
    train_r2_list.append(train_r2)
    train_rmse_list.append(train_rmse)

    test_mse_list.append(test_mse)
    test_mae_list.append(test_mae)
    test_r2_list.append(test_r2)
    test_rmse_list.append(test_rmse)

    # Store detailed predictions
    train_predictions_dict[f'fold_{fold + 1}'] = {
        'y_true': y_train.values,
        'y_pred': train_predictions,
        'indices': train_idx
    }

    test_predictions_dict[f'fold_{fold + 1}'] = {
        'y_true': y_test.values,
        'y_pred': test_predictions,
        'indices': test_idx
    }

    # Store fold details
    fold_details = {
        'fold': fold + 1,
        'test_samples': len(test_idx),
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    fold_details_list.append(fold_details)

    print(f"Fold {fold + 1} Performance:")
    print("  Training Set:")
    print(f"    MSE: {train_mse:.3f}, RMSE: {train_rmse:.3f}")
    print(f"    MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    print("  Test Set:")
    print(f"    MSE: {test_mse:.3f}, RMSE: {test_rmse:.3f}")
    print(f"    MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
    print("-" * 60)

fold_details_df = pd.DataFrame(fold_details_list)
fold_details_df.to_csv(RESULTS_PATH, index=False, encoding='utf-8-sig')
print(f"\nFold details saved to: {RESULTS_PATH}")


def calculate_stats(metric_list):
    """Calculate mean and standard deviation of a metric list."""
    return np.mean(metric_list), np.std(metric_list)

train_mean_mse, train_std_mse = calculate_stats(train_mse_list)
train_mean_mae, train_std_mae = calculate_stats(train_mae_list)
train_mean_r2, train_std_r2 = calculate_stats(train_r2_list)
train_mean_rmse, train_std_rmse = calculate_stats(train_rmse_list)

test_mean_mse, test_std_mse = calculate_stats(test_mse_list)
test_mean_mae, test_std_mae = calculate_stats(test_mae_list)
test_mean_r2, test_std_r2 = calculate_stats(test_r2_list)
test_mean_rmse, test_std_rmse = calculate_stats(test_rmse_list)

print("\n" + "=" * 80)
print("5-Fold Cross-Validation Performance Summary")
print("=" * 80)

print("\n=== Training Set Performance ===")
print(f"Mean MSE:  {train_mean_mse:.3f} ± {train_std_mse:.3f}")
print(f"Mean RMSE: {train_mean_rmse:.3f} ± {train_std_rmse:.3f}")
print(f"Mean MAE:  {train_mean_mae:.3f} ± {train_std_mae:.3f}")
print(f"Mean R²:   {train_mean_r2:.3f} ± {train_std_r2:.3f}")

print("\n=== Test Set Performance ===")
print(f"Mean MSE:  {test_mean_mse:.3f} ± {test_std_mse:.3f}")
print(f"Mean RMSE: {test_mean_rmse:.3f} ± {test_std_rmse:.3f}")
print(f"Mean MAE:  {test_mean_mae:.3f} ± {test_std_mae:.3f}")
print(f"Mean R²:   {test_mean_r2:.3f} ± {test_std_r2:.3f}")

# Plot overall test set predictions
all_test_true = np.concatenate([fold['y_true'] for fold in test_predictions_dict.values()])
all_test_pred = np.concatenate([fold['y_pred'] for fold in test_predictions_dict.values()])


# ✅ 将最终预测值插入到原数据第一列
data_with_pred = data.copy()
data_with_pred.insert(0, 'predicted_value', all_test_predictions_full)

# ✅ 保存结果
data_with_pred.to_csv(TEST_RESULT, index=False, encoding='utf-8-sig')
print("✅ Test predictions saved to test_result.csv")

# Train final model on full dataset
print("\nTraining final model...")
final_model = TabPFNRegressor(device=DEVICE)
final_model.fit(X, y)
joblib.dump(final_model, MODEL_PATH)


