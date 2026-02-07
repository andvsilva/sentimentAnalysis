###########################################################
### STEP 3 - Training and Prediction
###########################################################

# ==============================
# Imports
# ==============================
import time
import pickle
from datetime import datetime
from icecream import ic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.neural_network_torch import NeuralNetworkTorch
from models.trainer_torch import TorchTrainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    average_precision_score,
    f1_score
)
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import toolkit as tool
import os
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================
# Helper functions
# ==============================
def sentiment(label: float) -> str:
    feeling = ""
    """Map original score to binary sentiment."""
    if label in (4.0, 5.0):
        feeling = "0" # positive
    elif label in (1.0, 2.0):
        feeling = "1" # negative

    return feeling

# ==============================
# Start execution
# ==============================
start_time = time.time()
now = datetime.now()

print("date..............:", now)
print("Loading dataset - for modeling...")

# Load dataset
df_processed = pd.read_feather('../datasets/feather/featured.ftr')
df_processed['Sentiment'] = df_processed['Score'].apply(sentiment)

X_text = df_processed['Text']
y = df_processed['Sentiment']

tool.release_memory(df_processed)

# ==============================
# Class distribution (before SMOTE)
# ==============================
print('Class distribution BEFORE SMOTE:')
print(
    y.value_counts(normalize=True)
     .mul(100)
     .round(2)
     .map("{:.2f}%".format)
)


# ==============================
# Vectorization
# ==============================
print('Vectorizing text...')
cv = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X = cv.fit_transform(X_text)

# ==============================
# SMOTE balancing
# ==============================
print('Balancing dataset with SMOTE...')
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


print("X.shape =", X_res.shape)
print("y.shape =", y_res.shape)

print('Class distribution AFTER SMOTE:')
print(
    y_res.value_counts(normalize=True)
         .mul(100)
         .round(2)
         .map("{:.2f}%".format)
)


# ==============================
# Train / Validation split
# ==============================
x_train, x_validation, y_train, y_validation = train_test_split(
    X_res, y_res, test_size=0.2, random_state=0
)

# sparse → dense
x_train = x_train.toarray().astype("float32")
x_validation = x_validation.toarray().astype("float32")

y_train = y_train.astype("float32").to_numpy()
y_validation = y_validation.astype("float32").to_numpy()

tool.release_array(X)
tool.release_array(y)

path_models = 'models/'

os.chdir(path_models)

# ==============================
# Deep Learning model 
# ==============================
# tensors
X_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_val_t = torch.tensor(x_validation, dtype=torch.float32)
y_val_t = torch.tensor(y_validation, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

model = NeuralNetworkTorch(input_dim=X_train_t.shape[1])
trainer = TorchTrainer(model)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10
)

# ==============================
# Learning curves
# ==============================
pd.DataFrame(history)[['loss', 'val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
os.chdir('../')
plt.savefig('../pngs/learningcurves_torch.png')


# ==============================
# Evaluation — Deep Learning
# ==============================
y_pred = model.predict(x_validation)

print('Classification Report')
print(classification_report(y_validation, y_pred, target_names=['Positivo', 'Negativo']))

print('ROC AUC:', roc_auc_score(y_validation, y_pred))
print('Accuracy:', accuracy_score(y_validation, y_pred))
print('Avg Precision:', average_precision_score(y_validation, y_pred))
print('F1:', f1_score(y_validation, y_pred))

cm = confusion_matrix(y_validation, y_pred)
tool.plot_confusion_matrix(cm, classes=['Positivo', 'Negativo'])

# ==============================
# Baseline ML — GaussianNB
# ==============================
clf_gnb = GaussianNB()
clf_gnb.fit(x_train, y_train)

y_gnb = clf_gnb.predict(x_validation)
gnb_auc = roc_auc_score(y_validation, y_gnb)

model_probs = model.predict(x_validation)
model_auc = roc_auc_score(y_validation, model_probs)


# ==============================
# ROC Curves
# ==============================
fpr_gnb, tpr_gnb, _ = roc_curve(y_validation, y_gnb)
fpr_dl, tpr_dl, _ = roc_curve(y_validation, model_probs)

# Estilo seaborn
sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(16, 12))

# GaussianNB ROC
sns.lineplot(
    x=fpr_gnb,
    y=tpr_gnb,
    label=f"ML: GaussianNB (AUC={gnb_auc:.4f})"
)

# Neural Network ROC
sns.lineplot(
    x=fpr_dl,
    y=tpr_dl,
    label=f"DL: PyTorch (AUC={model_auc:.4f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.savefig("../pngs/model_ROC_curves_torch.png", dpi=300, bbox_inches="tight")
#plt.show()

# ==============================
# Save models
# ==============================
torch.save(model.state_dict(), "model_torch.pth")

with open("cv.pkl", "wb") as f:
    pickle.dump(cv, f)

print("Models saved successfully.")

# ==============================
# Execution time
# ==============================
time_exec_min = round((time.time() - start_time) / 60, 4)
print(f"Execution time: {time_exec_min} minutes")
print("All done. Good work.")