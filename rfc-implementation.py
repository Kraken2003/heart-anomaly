from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Convert torch tensors to numpy arrays
lead_data, label, age, sex = ds[0]
lead_data = lead_data.numpy()
label = label.numpy()
age = age.numpy()
sex = sex.numpy()

# Flatten lead data and concatenate age and sex features
lead_data_flattened = lead_data.reshape(lead_data.shape[0], -1)
features = np.hstack((lead_data_flattened, age.reshape(-1, 1), sex.reshape(-1, 1)))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# RandomForestClassifier implementation
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# XGBoost implementation
params = {
    'booster': 'gbtree',
    'n_estimators': 100,
    'eta': 0.1,
    'max_depth': 5,
    'lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

xgb_cl = xgb.XGBClassifier(**params)
xgb_cl.fit(X_train, y_train)
y_pred_xgb = xgb_cl.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)
