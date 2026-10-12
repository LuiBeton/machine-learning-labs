import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, RepeatedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# ==========================================
# 1. Завантаження даних та розрахунок складності
# ==========================================
print("--- 1. Завантаження даних та аналіз складності ---")
df = pd.read_csv('processed_tertyshnyk.csv')
X = df.drop(columns=['quality'])
y = df['quality']

p = X.shape[1] # Кількість початкових ознак (11)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

degrees = [1, 2, 3, 4, 5]
complexity_results = []

for d in degrees:
    # Точна формула комбінаторики: C(p+d, d) - 1
    n_features_theory = math.comb(p + d, d) - 1
    
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv, 
        scoring='neg_root_mean_squared_error', 
        return_train_score=True, n_jobs=-1
    )
    
    tr_rmse = -scores['train_score'].mean()
    cv_rmse = -scores['test_score'].mean()
    cv_std = (-scores['test_score']).std()
    se = cv_std / np.sqrt(5) # Standard Error
    
    complexity_results.append({
        'Degree (d)': d,
        'Features (n_poly)': n_features_theory,
        'Train RMSE': round(tr_rmse, 4),
        'CV RMSE': round(cv_rmse, 4),
        'CV Std': round(cv_std, 4),
        'SE': round(se, 4),
        'Gap': round(cv_rmse - tr_rmse, 4)
    })

comp_df = pd.DataFrame(complexity_results)
print(comp_df.to_string(index=False))

# Графік: Крива валідації
plt.figure(figsize=(7, 4))
plt.plot(comp_df['Degree (d)'], comp_df['Train RMSE'], 'o-', label='Train RMSE', color='blue')
plt.errorbar(comp_df['Degree (d)'], comp_df['CV RMSE'], yerr=comp_df['CV Std'], 
             fmt='o-', label='CV RMSE ± Std', color='orange', capsize=4)
plt.xlabel('Степінь полінома (d)')
plt.ylabel('RMSE (менше — краще)')
plt.title('Крива валідації для поліноміальної складності')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 2. Криві навчання (Learning Curves)
# ==========================================
print("\n--- 2. Побудова кривих навчання ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, d in enumerate([1, 2]):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X_train, y_train, cv=cv, 
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    
    tr_mean = -np.mean(train_scores, axis=1)
    te_mean = -np.mean(test_scores, axis=1)
    
    ax = axes[idx]
    ax.plot(train_sizes, tr_mean, 'o-', color='blue', label='Train RMSE')
    ax.plot(train_sizes, te_mean, 's-', color='orange', label='CV RMSE')
    ax.set_xlabel('Розмір навчальної вибірки (n)')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Крива навчання для d={d}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 3. Підбір гіперпараметрів регуляризації (на d=2)
# ==========================================
print("\n--- 3. Підбір alpha для регуляризованих моделей (на d=2) ---")
alphas = np.logspace(-3, 2, 20)
l1_ratios = [0.2, 0.5, 0.8]

# 3.1 Ridge
ridge_cv = [ -cross_validate(Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('regressor', Ridge(alpha=a))]), X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)['test_score'].mean() for a in alphas ]
best_ridge_a = alphas[np.argmin(ridge_cv)]

# 3.2 Lasso
lasso_cv = [ -cross_validate(Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('regressor', Lasso(alpha=a, max_iter=200000, tol=1e-3))]), X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)['test_score'].mean() for a in alphas ]
best_lasso_a = alphas[np.argmin(lasso_cv)]

# 3.3 ElasticNet
best_en_score = float('inf')
best_en_a = None
best_en_l1 = None

for l1 in l1_ratios:
    for a in alphas:
        score = -cross_validate(Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('regressor', ElasticNet(alpha=a, l1_ratio=l1, max_iter=200000, tol=1e-3))]), X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)['test_score'].mean()
        if score < best_en_score:
            best_en_score = score
            best_en_a = a
            best_en_l1 = l1

print(f"Найкраще Ridge alpha: {best_ridge_a:.4f} (CV RMSE: {min(ridge_cv):.4f})")
print(f"Найкраще Lasso alpha: {best_lasso_a:.4f} (CV RMSE: {min(lasso_cv):.4f})")
print(f"Найкраще ElasticNet alpha: {best_en_a:.4f}, l1_ratio: {best_en_l1} (CV RMSE: {best_en_score:.4f})")

# Графік сили регуляризації
plt.figure(figsize=(8, 4))
plt.plot(alphas, ridge_cv, 'o-', label='Ridge CV RMSE', color='blue')
plt.plot(alphas, lasso_cv, 's-', label='Lasso CV RMSE', color='green')
plt.xscale('log')
plt.xlabel('Сила регуляризації (alpha)')
plt.ylabel('CV RMSE')
plt.title('Вплив alpha на валідаційну похибку (d=2)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 4. Регуляризаційні шляхи (Regularization Paths)
# ==========================================
print("\n--- 4. Регуляризаційні шляхи коефіцієнтів ---")
X_poly_tr = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
X_poly_tr_scaled = StandardScaler().fit_transform(X_poly_tr)

ridge_coefs = []
lasso_coefs = []

for a in alphas:
    r_m = Ridge(alpha=a).fit(X_poly_tr_scaled, y_train)
    l_m = Lasso(alpha=a, max_iter=200000, tol=1e-3).fit(X_poly_tr_scaled, y_train)
    ridge_coefs.append(r_m.coef_)
    lasso_coefs.append(l_m.coef_)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(alphas, ridge_coefs)
axes[0].set_xscale('log')
axes[0].set_xlabel('alpha')
axes[0].set_ylabel('Коефіцієнти')
axes[0].set_title('Регуляризаційний шлях: Ridge (L2)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(alphas, lasso_coefs)
axes[1].set_xscale('log')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('Коефіцієнти')
axes[1].set_title('Регуляризаційний шлях: Lasso (L1)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 5. Перевірка стійкості (RepeatedKFold)
# ==========================================
print("\n--- 5. Перевірка стійкості вибору гіперпараметрів (RepeatedKFold) ---")
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

rkf_scores = cross_validate(
    Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('regressor', Ridge(alpha=best_ridge_a))]),
    X_train, y_train, cv=rkf, scoring='neg_root_mean_squared_error', n_jobs=-1
)

rkf_rmse_mean = -rkf_scores['test_score'].mean()
rkf_rmse_std = (-rkf_scores['test_score']).std()
print(f"Стійкість Ridge(alpha={best_ridge_a:.4f}) на RepeatedKFold: Mean CV RMSE = {rkf_rmse_mean:.4f} ± {rkf_rmse_std:.4f}")

# ==========================================
# 6. Фінальне оцінювання на test set
# ==========================================
print("\n--- 6. Фінальне оцінювання вибраної моделі на тестовій вибірці ---")
# ВАЖЛИВО: Використовуємо САМЕ ТЕ alpha, яке визначила крос-валідація!
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=best_ridge_a))
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
r2_test = r2_score(y_test, y_pred)

print(f"Зафіксоване alpha:              {best_ridge_a:.4f}")
print(f"Середня абсолютна похибка (MAE): {mae_test:.4f}")
print(f"Квадратична похибка (RMSE):       {rmse_test:.4f}")
print(f"Коефіцієнт детермінації (R2):    {r2_test:.4f}")
