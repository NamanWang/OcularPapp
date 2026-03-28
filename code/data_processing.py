# %%
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # Used for saving and loading StandardScaler objects

# Load data
data = pd.read_csv('drug_1218.csv')

# Separate labels, raw information, and features
y = data['value']
meta = data.iloc[:, :5]   # First 16 columns: raw information
X = data.iloc[:, 6:]      # From column 17 to the last column (features)
print(X.columns)


# ===== Step 3: Feature scaling =====
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save the StandardScaler object for future prediction use
joblib.dump(scaler, 'scaler.pkl')
print("StandardScaler has been saved as 'scaler.pkl'")

# ===== Merge scaled features with raw information and labels =====
data_after_scaling = pd.concat([meta.reset_index(drop=True),
                                y.reset_index(drop=True),
                                X_scaled.reset_index(drop=True)], axis=1)

# Save the scaled data
data_after_scaling.to_csv('drug_1218_scaling.csv', index=False)
print("Data after scaling shape:", data_after_scaling.shape)