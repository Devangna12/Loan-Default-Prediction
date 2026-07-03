# ==========================================
# Loan price  - DATA PREPROCESSING
# ==========================================

import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plot  
import seaborn as sus  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split  
import joblib

# ===== STEP 1: LOAD DATA =====

raw_data = pd.read_csv(r"C:\Users\Devangna\OneDrive\Desktop\ds\python\credit_risk_dataset.csv")

data = pd.DataFrame(raw_data)

print("\n" + "="*60)
print("STEP 1: CHECKING FOR MISSING VALUES")
print("="*60)

print("\nMissing value count per column:")
print(data.isnull().sum())

# ===== STEP 2: REMOVE COLUMNS WITH TOO MANY MISSING VALUES =====
data_null_clm = data.isnull().sum() / data.shape[0] * 100  

print("\nMissing value percentage per column:")
print(data_null_clm)

# Find columns where more than 20% of data is missing
null20_clm_list = data_null_clm[data_null_clm > 20].index

# Remove columns with more than 20% missing data (too much data lost)
data_drop_clm = data.drop(columns=null20_clm_list)

print(f"\nRemoved {len(null20_clm_list)} columns with >20% missing values")
print("Remaining missing values:")
print(data_drop_clm.isnull().sum())

# ===== STEP 3: SEPARATE DATA BY TYPE =====

print("\nData shapes:")
print(f"Original data shape: {data.shape}") 
print(f"After dropping columns with >20% missing: {data_drop_clm.shape}")

data_object = data_drop_clm.select_dtypes(include=['object'])
print(f"Categorical (text) columns shape: {data_object.shape}")

data_numaric = data_drop_clm.select_dtypes(include=['int', 'float'])
print(f"Numerical columns shape: {data_numaric.shape}")

# ===== STEP 4: HANDLE MISSING VALUES IN NUMERICAL COLUMNS =====

print("\n" + "="*60)
print("STEP 2: FILLING MISSING VALUES")
print("="*60)

print("\nFilling numerical columns with MEDIAN:")
numeic_clo = data_numaric.isnull().sum()
print(data_numaric.isnull().sum())

for i in numeic_clo.index:
    data_numaric[i].fillna(data_numaric[i].median(), inplace=True)

print("All missing numerical values filled with median ")
print(data_numaric.isnull().sum())

# ===== STEP 5: HANDLE MISSING VALUES IN CATEGORICAL COLUMNS =====

print("\nFilling categorical columns with MODE (most common value):")
string_clo = data_object.isnull().sum()

for i in string_clo.index:
    data_object[i].fillna(data_object[i].mode()[0], inplace=True)

print(f"All missing categorical values filled")
print(f"Total remaining missing values: {data_object.isnull().sum().sum()}")
data_encoded = data_object.copy()

print(data_object.head())
# ============================================================
# SECTION 2: ENCODE OBJECT (CATEGORICAL) COLUMNS (SECOND)
# ============================================================

print("\n" + "="*60)
print("STEP 5: ENCODING CATEGORICAL COLUMNS")
print("="*60)
data_encoded = data_object.copy()
print(data_encoded['Education_Level'].value_counts())
print(data_encoded['Housing_Status'].value_counts())

# ============================================================
# SECTION 1: ORDINAL ENCODING 
# ============================================================

print("\n" + "="*60)
print("ENCODING TYPE 1: ORDINAL ENCODING (Quality/Condition Levels)")
print("="*60)

# ordinal_mappings = {
#     'Education_Level': {'High School ': 1, 'Bachelors ': 2, 'Masters ': 3, 'PhD ': 4},  
#     'Housing_Status': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
# }

# print("\n Applying Ordinal Encoding:")
# for col, mapping in ordinal_mappings.items():
#     # Check if column exists in our data
#     if col in data_encoded.columns:
#         # Use .map() to replace values: 'Fa'→1, 'TA'→2, etc.
#         data_encoded[col] = data_encoded[col].map(mapping)
#         print(f"   {col} → {mapping}")



education_mapping = {'High School': 1,'Bachelors': 2,'Masters': 3,'PhD': 4}

data_encoded['Education_Level'] = data_encoded['Education_Level'].map(education_mapping)

print("\nEducation_Level after ordinal encoding:")
print(data_encoded['Education_Level'].value_counts())

#  One-Hot Encoding for Housing_Status
data_encoded = pd.get_dummies(data_encoded,columns=['Housing_Status'],drop_first=True,dtype='int')

print("\nHousing_Status after one-hot encoding:")
print(data_encoded.head())

# ============================================================
# SECTION 3: Train_split_test
# ============================================================

final_data = pd.concat([data_numaric, data_encoded], axis=1)
print(final_data.head())

print(final_data.shape)

X = final_data.drop(columns=['Default'])
y = final_data['Default']



from sklearn.model_selection import train_test_split
x_train, x_test, y_train_target, y_test_target = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(x_train.columns)
# ============================================================
# SECTION 4: Scaling
# ============================================================

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit scaler on training data only
scaled_data_final_X_tarin_arr = scaler.fit_transform(x_train)
scaled_data_final_X_tarin = pd.DataFrame(scaled_data_final_X_tarin_arr, columns=x_train.columns)
# Transform test data using the same scaler (do not fit again)
scaled_data_final_X_test_arr = scaler.transform(x_test)
scaled_data_final_X_test = pd.DataFrame(scaled_data_final_X_test_arr, columns=x_test.columns)
print(scaled_data_final_X_tarin.shape)
print(scaled_data_final_X_test.shape)
print(scaled_data_final_X_tarin.describe())
print(scaled_data_final_X_test.describe())
# ============================================================
# SECTION 4: lodistictic model
# ============================================================
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(scaled_data_final_X_tarin,y_train_target)
y_predict=lr.predict(scaled_data_final_X_test)
y_score=lr.score(scaled_data_final_X_test,y_test_target)
print(y_score)
from sklearn.metrics import precision_score,recall_score,precision_recall_fscore_support,f1_score,classification_report,confusion_matrix
print('p_S',precision_score(y_test_target,y_predict))
print('r_s',recall_score(y_test_target,y_predict))
print('r1_socre',f1_score(y_test_target,y_predict))
print("---"*15)
print(confusion_matrix(y_test_target,y_predict))
print("---"*15)
print(classification_report(y_test_target,y_predict))
# Save trained model and scaler for use in the frontend
joblib.dump(lr, "loan_model.pkl")
joblib.dump(scaler, "loan_scaler.pkl")
print("Saved model to loan_model.pkl and scaler to loan_scaler.pkl")
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline

# # Build a pipeline: Polynomial expansion + Logistic Regression
# model = Pipeline([
#     ("poly", PolynomialFeatures(degree=2, include_bias=False)),
#     ("logreg", LogisticRegression(class_weight='balanced', random_state=42))
# ])

# # Fit the model on training data
# model.fit(scaled_data_final_X_tarin, y_train_target)

# # Predict on test data
# y_predict = model.predict(scaled_data_final_X_test)

# # Evaluate
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test_target, y_predict))
# print(classification_report(y_test_target, y_predict))
