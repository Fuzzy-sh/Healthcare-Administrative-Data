import pickle
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
# Get root path dynamically
# ROOT_DIR = os.path.dirname(os.path.abspath(""))
# sys.path.append(ROOT_DIR)  # Adds the project root to PYTHONPATH
# print(ROOT_DIR)

from utils import PreprocessData
from config.config_param import config_param





def group_treatment(df):

    # Define groups
    medications = ['a:antidepressant_medication', 'a:antipsychotic_medication', 
                'a:benzo_medication', 'a:mood_medication']
    
    therapy = ['a:mental_health_therapy', 'a:occupational_therapy']

    counseling = ['a:rehabilitation', 'a:addictions_counselling', 'a:unspecified_counselling']

    return df[medications].apply(lambda x: 1 if x.sum()>0 else 0, axis=1), df[therapy].apply(lambda x: 1 if x.sum()>0 else 0, axis=1), df[counseling].apply(lambda x: 1 if x.sum()>0 else 0, axis=1)


config_params = config_param()
param_dic = config_params.get_params_updated()
pred = PreprocessData(param_dic)

test_path =  param_dic['split_files']['test']

test_data = pred.read_data(test_path)
# # Train KNN Model
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# # 🔹 Save KNN Model
# with open("knn_model.pkl", "wb") as f:
#     pickle.dump(knn, f)
print ("loading test data")
feature_columns = [col for col in test_data.columns if col.startswith('o:')]
action_columns = [col for col in test_data.columns if col.startswith('a:')]

print ("grouping treatment")
action_columns_grop = ['a:medical_treatment', 'a:therapy', 'a:counseling']
test_data['a:medical_treatment'], test_data['a:therapy'], test_data['a:counseling'] = group_treatment(test_data)

print ("getting values")
X_test = test_data[feature_columns].values
y_test = test_data[action_columns].values
y_test_group = test_data[action_columns_grop].values
# 🔹 Load KNN Model
# with open("../results_knn/knn_20.pkl", "rb") as f:


file_path = "/work/messier_lab/fuzzy/results_knn/"
file_path = "results_knn/"
print ("loading knn mdoel")
with open(file_path+"knn_191.pkl", "rb") as f:
    knn_loaded = pickle.load(f)

print ("loading knn group mdoel")
with open(file_path+"knn_group_191.pkl", "rb") as g:
    knn_loaded_group = pickle.load(g)


# ✅ Test Loaded Model
print ("predicting knn")
y_pred = knn_loaded.predict(X_test)
print ("predicting knn group")
y_pred_group = knn_loaded_group.predict(X_test)

# # Get probability estimates
# # Get probability estimates (ensures shape is valid)
# print ("predicting")
# probs = knn_loaded.predict_proba(X_test)  # Shape should be (N_samples, N_classes)
# probs_group = knn_loaded_group.predict_proba(X_test)  # Shape should be (N_samples, N_classes)
# # Convert list of arrays to a single NumPy array
# probs = np.array(probs[0], dtype=object)  # Keep object type to handle irregular shapes
# probs_group = np.array(probs_group, dtype=object)  # Keep object type to handle irregular shapes
# print("Shape of probs:", probs.shape)  # Debugging: Check if it's (N_samples, N_classes)
# print("Shape of probs:", probs_group.shape)  # Debugging: Check if it's (N_samples, N_classes)

# # Compute uncertainty (1 - max probability of predicted class)
# uncertainty = 1 - np.max(probs, axis=1)
# # Compute uncertainty (1 - max probability of predicted class)
# uncertainty_group = 1 - np.max(probs_group, axis=1)

# print("Uncertainty scores:", uncertainty[:10])  # Show first 10 uncertainty values

# # Compute uncertainty (1 - max probability of predicted class)
# uncertainty_group = 1 - np.max(probs_group, axis=1)

# # print("Uncertainty scores_group:", uncertainty_group[:10])  # Show first 10 uncertainty values



precision_micro = precision_score(y_test, y_pred, average='micro')  # Micro-averaged precision
precision_weighted = precision_score(y_test, y_pred, average='weighted')  # Weighted by class size
recall_micro = recall_score(y_test, y_pred, average='micro')  # Micro-averaged recall
recall_weighted = recall_score(y_test, y_pred, average='weighted')  # Weighted by class size
f1_micro = f1_score(y_test, y_pred, average='micro')  # Micro-averaged F1
f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Weighted by class size


print(f"Micro Precision: {precision_micro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Micro Recall: {recall_micro:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Micro F1: {f1_micro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")


precision_micro = precision_score(y_test_group, y_pred, average='micro')  # Micro-averaged precision
precision_weighted = precision_score(y_test_group, y_pred, average='weighted')  # Weighted by class size
recall_micro = recall_score(y_test_group, y_pred, average='micro')  # Micro-averaged recall
recall_weighted = recall_score(y_test_group, y_pred, average='weighted')  # Weighted by class size
f1_micro = f1_score(y_test_group, y_pred, average='micro')  # Micro-averaged F1
f1_weighted = f1_score(y_test_group, y_pred, average='weighted')  # Weighted by class size


print(f"Micro Precision: {precision_micro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Micro Recall: {recall_micro:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Micro F1: {f1_micro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")



