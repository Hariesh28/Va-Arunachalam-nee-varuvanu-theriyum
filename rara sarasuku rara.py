import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🚀 Load dataset (you can replace this with your own CSV)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 🔍 Step 1: Feature Selection (Chi-Squared Test)
selector = SelectKBest(score_func=chi2, k=20)  # top 20 features
X_selected = selector.fit_transform(X, y)

# 🎯 Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# ⚖️ Step 3: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔻 Step 4: Apply PCA after scaling
pca = PCA(n_components=0.95, random_state=42)  # retain 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")

# 🧠 Step 5: Define MLP Classifier and Hyperparameter Grid
mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(128, 64), (100, 50), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

# 🧪 Step 6: GridSearch for Best Parameters
grid = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train_pca, y_train)

# 🏆 Step 7: Best Model Evaluation
best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test_pca)

print("\n✅ Best Parameters:", grid.best_params_)
print("🎯 Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 📊 Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 📈 Step 9: Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(best_mlp.loss_curve_, label='Training Loss', color='green')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

'''
DEI PARAMA PADI DA
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 📥 1. Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ✂️ 2. Feature Selection
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)

# 📊 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# ⚙️ 4. Build a Full Pipeline
pipeline = Pipeline([
    ('scaler', RobustScaler()),                       # better scaling for outliers
    ('power', PowerTransformer()),                    # makes data Gaussian-like
    ('pca', PCA(n_components=0.95, random_state=42)), # preserve 95% variance
    ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, random_state=42))
])

# 🧪 5. Hyperparameter Grid
param_grid = {
    'mlp__hidden_layer_sizes': [(128, 64), (256, 128, 64)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate': ['constant', 'adaptive']
}

# 🔁 6. Cross-validation and grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# 🏆 7. Evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Best Hyperparameters:", grid.best_params_)
print("🎯 Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 🔥 8. Confusion Matrix (with % labels)
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='Blues', values_format='.2%')
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

# 📈 9. Learning Curve
plt.figure(figsize=(8,5))
plt.plot(best_model.named_steps['mlp'].loss_curve_, color='orange', label='Training Loss')
plt.title("MLP Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


'''
Sollunga Mamakutty
Enna Intha Timela
Call Pannirukinka
LongDrive Polamaa

Aama la
Pona Wednesday Ponathu
Intha Wednesday Vanthuruchu
Aanaa Etho Oru Maasam
Aana Maathiri Irukkulla

Aanaa Pona Thadavaiye
Pradeep Kitta Reason Solla
Evlo Kastamaa Irunthuchu Theriyumaa

Hey Pradeep
Innaiku Un Call a Ennala
Attend Panna Mudiyathu Da
En Aththimbel Oorla Irunthu
Familyoda Varraram
Enka Appa Avanka Ellaraiyum Enkayaathu
Velila Koottindu Poitu Vaa nu
Sollinte Irukkaru Da

Its Ok Baby..
No Problem...Poittu Vaanka...

Kaduppa Irukkuda Baby..
Enakke Innaiku Oru Naal Than Leave
Unna Paakkalam nu Irunthen
Athulaium Avaru Vanthu Avankala
Inka Kootindu Poitu Vaa
Anka Kootindu Poitu vaa nu
Paduthuraru Da

Oru Naal Thaana Baby
Palla Kadichitu Poitu Vaanka Paathukalam
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Separate features and target
X = train.drop(columns=['target'])  # Replace 'target' with actual label column
y = train['target']
X_test_final = test.copy()

# 3. Fill nulls
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test_final)

# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 5. Feature selection (top k best features)
selector = SelectKBest(score_func=f_classif, k=30)  # You can tune k
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

# 6. PCA for dimensionality reduction
pca = PCA(n_components=20)  # You can tune this as well
X_pca = pca.fit_transform(X_selected)
X_test_pca = pca.transform(X_test_selected)

# 7. Split training for validation
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

# 8. Define model and hyperparameter grid
mlp = MLPClassifier(max_iter=1000, random_state=42)
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (128, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'solver': ['adam']
}

# 9. GridSearchCV for best model
grid = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# 10. Evaluate on validation
y_pred = grid.predict(X_val)
print("Best Parameters:", grid.best_params_)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# 11. Refit on full data for submission
grid.best_estimator_.fit(X_pca, y)

# 12. Predict on test
y_test_pred = grid.predict(X_test_pca)

# 13. Save submission
submission = pd.DataFrame({'id': test['id'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
