import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, make_moons
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# 1. Геометричний експеримент (make_moons)
# ==========================================
X_m, y_m = make_moons(n_samples=500, noise=0.2, random_state=42)
X_m[:, 1] *= 18.0  # Штучно спотворюємо масштаб 2-ї ознаки

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
models_moons = {
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Linear SVM': SVC(kernel='linear', C=1),
    'RBF SVM': SVC(kernel='rbf', C=1, gamma='scale')
}

for i, (scale_name, scaler) in enumerate([("Без масштабування", None), ("З StandardScaler", StandardScaler())]):
    for j, (m_name, model) in enumerate(models_moons.items()):
        if scaler:
            pipe = Pipeline([('scaler', scaler), ('model', model)])
        else:
            pipe = Pipeline([('model', model)])
        
        pipe.fit(X_m, y_m)
        f1 = f1_score(y_m, pipe.predict(X_m))
        
        ax = axes[i, j]
        xx, yy = np.meshgrid(np.linspace(X_m[:, 0].min()-1, X_m[:, 0].max()+1, 100),
                             np.linspace(X_m[:, 1].min()-10, X_m[:, 1].max()+10, 100))
        Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X_m[:, 0], X_m[:, 1], c=y_m, cmap='coolwarm', edgecolors='k', s=20)
        ax.set_title(f"{m_name}\n({scale_name}) F1={f1:.3f}")

plt.tight_layout()
plt.show()

# ==========================================
# 2. Завантаження Mushroom Dataset та підготовка
# ==========================================
data = fetch_openml(name='mushroom', version=1, as_frame=True, parser='auto')
X = data.data
y = (data.target == 'p').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 3. Налаштування та порівняння моделей
# ==========================================
pipelines = {
    "Nearest Centroid": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', MinMaxScaler()),
        ('model', NearestCentroid())
    ]),
    "kNN (k=5, distance)": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', MinMaxScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'))
    ]),
    "Linear SVM": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', MinMaxScaler()),
        ('model', SVC(kernel='linear', C=1, random_state=42))
    ]),
    "RBF SVM": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', MinMaxScaler()),
        ('model', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
    ]),
    "Logistic Regression (Reference)": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
}

cv_results = []
for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['f1', 'accuracy'])
    cv_results.append({
        "Модель": name,
        "CV F1": round(scores['test_f1'].mean(), 4),
        "CV Std": round(scores['test_f1'].std(), 4),
        "CV Accuracy": round(scores['test_accuracy'].mean(), 4)
    })

cv_df = pd.DataFrame(cv_results)
print("\n--- Порівняння моделей на 5-fold крос-валідації ---")
print(cv_df.to_string(index=False))

# ==========================================
# 4. Фінальна оцінка обраної моделі (RBF SVM) на тестовій вибірці
# ==========================================
best_model = pipelines["RBF SVM"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n--- Фінальна оцінка обраної моделі (RBF SVM) на тестовій вибірці ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
