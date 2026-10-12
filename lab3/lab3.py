import warnings
warnings.filterwarnings('ignore')

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml, make_moons
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, pairwise_distances

RANDOM_STATE = 42

# ==========================================
# 1. Контрольний геометричний експеримент (make_moons)
# ==========================================
print("--- 1. Геометричний експеримент (make_moons) ---")
X_m, y_m = make_moons(n_samples=500, noise=0.2, random_state=RANDOM_STATE)
X_m[:, 1] *= 18.0  # Спотворюємо масштаб

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_m, y_m, test_size=0.2, stratify=y_m, random_state=RANDOM_STATE
)

models_moons = {
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Linear SVM': SVC(kernel='linear', C=1, random_state=RANDOM_STATE),
    'RBF SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, (scale_name, scaler) in enumerate([("Без масштабування", None), ("StandardScaler", StandardScaler())]):
    for j, (m_name, model) in enumerate(models_moons.items()):
        if scaler:
            pipe = Pipeline([('scaler', scaler), ('model', model)])
        else:
            pipe = Pipeline([('model', model)])
        
        pipe.fit(X_m_train, y_m_train)
        y_m_pred = pipe.predict(X_m_test)
        f1_test = f1_score(y_m_test, y_m_pred, average='macro')
        
        ax = axes[i, j]
        xx, yy = np.meshgrid(np.linspace(X_m[:, 0].min()-1, X_m[:, 0].max()+1, 150),
                             np.linspace(X_m[:, 1].min()-10, X_m[:, 1].max()+10, 150))
        Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X_m_test[:, 0], X_m_test[:, 1], c=y_m_test, cmap='coolwarm', edgecolors='k', s=20)
        
        # Позначення опорних векторів для SVM
        if 'SVM' in m_name:
            svc_obj = pipe.named_steps['model']
            svs = svc_obj.support_vectors_
            if scaler:
                svs = pipe.named_steps['scaler'].inverse_transform(svs)
            ax.scatter(svs[:, 0], svs[:, 1], s=50, facecolors='none', edgecolors='purple', linewidths=1.2)
            
        ax.set_title(f"{m_name}\n({scale_name}) F1={f1_test:.3f}")
plt.tight_layout()
plt.show()  # Відкриває вікно з графіком!

# ==========================================
# 2. Підготовка індивідуальних даних
# ==========================================
print("\n--- 2. Завантаження Mushroom Dataset ---")
data = fetch_openml(name='mushroom', version=1, as_frame=True, parser='auto')
X = data.data
y = (data.target == 'p').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ==========================================
# 3. Дослідження масштабування (Scaling Study)
# ==========================================
print("\n--- 3. Дослідження масштабування (Scaling Study) ---")
scalers = {
    'none': 'passthrough',
    'standard': StandardScaler(),
    'minmax': MinMaxScaler()
}

scaling_models = {
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Linear SVM': SVC(kernel='linear', C=1, random_state=RANDOM_STATE),
    'RBF SVM': SVC(kernel='rbf', C=1, gamma='scale', random_state=RANDOM_STATE)
}

scaling_results = []
for m_name, model in scaling_models.items():
    for s_name, scaler in scalers.items():
        pipe = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('scaler', scaler),
            ('model', model)
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        scaling_results.append({
            'Model': m_name,
            'Scaler': s_name,
            'CV F1 Mean': scores['test_score'].mean(),
            'CV F1 Std': scores['test_score'].std()
        })

scaling_df = pd.DataFrame(scaling_results)
print(scaling_df.to_string(index=False))

# ==========================================
# 4. Пошук гіперпараметрів (GridSearchCV)
# ==========================================
print("\n--- 4. Пошук гіперпараметрів через GridSearchCV ---")

# kNN Grid Search
knn_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', MinMaxScaler()),
    ('model', KNeighborsClassifier())
])
knn_grid = GridSearchCV(
    knn_pipe,
    param_grid={
        'model__n_neighbors': [1, 3, 5, 7],
        'model__p': [1, 2],
        'model__weights': ['uniform', 'distance']
    },
    cv=cv, scoring='f1_macro', n_jobs=-1
)
knn_grid.fit(X_train, y_train)

# RBF SVM Grid Search
rbf_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', MinMaxScaler()),
    ('model', SVC(kernel='rbf', random_state=RANDOM_STATE))
])
rbf_grid = GridSearchCV(
    rbf_pipe,
    param_grid={
        'model__C': [0.1, 1, 10],
        'model__gamma': ['scale', 0.01, 0.1, 1.0]
    },
    cv=cv, scoring='f1_macro', n_jobs=-1
)
rbf_grid.fit(X_train, y_train)

# Heatmap для RBF SVM
heatmap_data = pd.DataFrame(rbf_grid.cv_results_)[['param_model__C', 'param_model__gamma', 'mean_test_score']]
heatmap_matrix = heatmap_data.pivot(index='param_model__gamma', columns='param_model__C', values='mean_test_score')
plt.figure(figsize=(7, 5))
sns.heatmap(heatmap_matrix, annot=True, fmt=".4f", cmap="viridis")
plt.title("Heatmap: C x Gamma для RBF SVM")
plt.tight_layout()
plt.show()  # Відкриває вікно з тепловою картою!

# ==========================================
# 5. Контрольований експеримент із розмірністю
# ==========================================
print("\n--- 5. Контрольований експеримент із розмірністю ---")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_tr_enc = encoder.fit_transform(X_train)

rng = np.random.default_rng(RANDOM_STATE)
dim_results = []
noise_counts = [0, 64, 256, 512]

for noise_c in noise_counts:
    if noise_c > 0:
        noise = rng.uniform(0.0, 1.0, size=(len(X_tr_enc), noise_c))
        X_aug = np.hstack([X_tr_enc, noise])
    else:
        X_aug = X_tr_enc.copy()
    
    # Обчислення d_min / d_max на скомплексованій підмножині (для швидкості)
    sub_sample = X_aug[:300]
    dists = pairwise_distances(sub_sample)
    np.fill_diagonal(dists, np.inf)
    d_min = dists.min(axis=1)
    np.fill_diagonal(dists, -np.inf)
    d_max = dists.max(axis=1)
    ratio_median = np.median(d_min / d_max)
    
    # Швидка оцінка kNN
    knn_test_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', KNeighborsClassifier(n_neighbors=5))])
    scores = cross_validate(knn_test_pipe, X_aug[:1500], y_train[:1500], cv=cv, scoring='f1_macro', n_jobs=-1)
    
    dim_results.append({
        'Noise Features': noise_c,
        'd_min/d_max Ratio': ratio_median,
        'kNN CV F1': scores['test_score'].mean()
    })

dim_df = pd.DataFrame(dim_results)
print(dim_df.to_string(index=False))

# Графік розмірності
plt.figure(figsize=(7, 4))
plt.plot(dim_df['Noise Features'], dim_df['d_min/d_max Ratio'], 'o-', label='d_min/d_max Ratio', color='red')
plt.plot(dim_df['Noise Features'], dim_df['kNN CV F1'], 's-', label='kNN CV F1', color='blue')
plt.xlabel("Кількість шумових ознак")
plt.ylabel("Значення метрики")
plt.title("Вплив прокляття розмірності на kNN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  # Відкриває графік розмірності!

# ==========================================
# 6. Порівняння моделей та Інженерний компроміс
# ==========================================
print("\n--- 6. Порівняння кандидатних моделей та Інженерний компроміс ---")
final_candidates = {
    'Nearest Centroid': Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)), ('scaler', MinMaxScaler()), ('model', NearestCentroid())]),
    'Best kNN': knn_grid.best_estimator_,
    'Linear SVM': Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)), ('scaler', MinMaxScaler()), ('model', SVC(kernel='linear', C=1, random_state=RANDOM_STATE))]),
    'Best RBF SVM': rbf_grid.best_estimator_,
    'Logistic Regression': Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)), ('scaler', MinMaxScaler()), ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
}

eng_summary = []
for name, pipe in final_candidates.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    pipe.fit(X_train, y_train)
    
    t_pred_start = time.time()
    pipe.predict(X_test)
    pred_time_ms = (time.time() - t_pred_start) * 1000
    
    joblib.dump(pipe, "temp_model.joblib")
    size_kb = os.path.getsize("temp_model.joblib") / 1024
    if os.path.exists("temp_model.joblib"):
        os.remove("temp_model.joblib")
        
    model_obj = pipe.named_steps['model']
    if hasattr(model_obj, 'support_vectors_'):
        n_saved = len(model_obj.support_vectors_)
    elif hasattr(model_obj, 'centroids_'):
        n_saved = len(model_obj.centroids_)
    elif hasattr(model_obj, '_fit_X'):
        n_saved = len(model_obj._fit_X)
    else:
        n_saved = len(model_obj.coef_.ravel())
        
    eng_summary.append({
        'Model': name,
        'CV F1': round(scores['test_score'].mean(), 4),
        'CV Std': round(scores['test_score'].std(), 4),
        'Pred Time (ms)': round(pred_time_ms, 2),
        'Size (KB)': round(size_kb, 1),
        'Saved Objects/SVs': n_saved
    })

eng_df = pd.DataFrame(eng_summary)
print(eng_df.to_string(index=False))

# ==========================================
# 7. Out-of-Fold Аналіз розбіжностей
# ==========================================
print("\n--- 7. Out-of-Fold Аналіз розбіжностей (kNN vs SVM) ---")
knn_oof = cross_val_predict(knn_grid.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
svm_oof = cross_val_predict(rbf_grid.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)

disagree_idx = np.flatnonzero(knn_oof != svm_oof)
print(f"Знайдено розбіжностей між kNN та RBF SVM: {len(disagree_idx)}")

# ==========================================
# 8. Фінальне оцінювання на test set
# ==========================================
print("\n--- 8. Фінальна оцінка обраної моделі (RBF SVM) на тестовій вибірці ---")
best_model = rbf_grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
