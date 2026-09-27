import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Завантаження даних та формування вибірок
df = pd.read_csv('processed_tertyshnyk.csv')

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Оцінка кількості ознак та побудова кривої валідації
degrees = [1, 2, 3] # Для 11 ознак степені >3 викликають вибух розмірності
complexity_results = []

for d in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    scores = cross_validate(
        model, X_train, y_train, cv=cv, 
        scoring='neg_root_mean_squared_error', 
        return_train_score=True
    )
    
    train_rmse = -scores['train_score'].mean()
    cv_rmse = -scores['test_score'].mean()
    cv_std = (-scores['test_score']).std()
    
    # Кількість ознак розраховується за формулою комбінаторики
    n_features = PolynomialFeatures(degree=d, include_bias=False).fit_transform(X_train).shape[1]
    
    complexity_results.append({
        'Степінь (d)': d,
        'Кількість ознак': n_features,
        'Train RMSE': train_rmse,
        'CV RMSE': cv_rmse,
        'CV Std': cv_std,
        'Gap': cv_rmse - train_rmse
    })

complexity_df = pd.DataFrame(complexity_results)
print("--- Дослідження складності полінома ---")
print(complexity_df.to_string(index=False))

# Графік кривої валідації
plt.figure(figsize=(7, 4))
plt.plot(complexity_df['Степінь (d)'], complexity_df['Train RMSE'], 'o-', label='Train RMSE', color='blue')
plt.errorbar(complexity_df['Степінь (d)'], complexity_df['CV RMSE'], yerr=complexity_df['CV Std'], 
             fmt='o-', label='CV RMSE ± Std', color='orange', capsize=4)
plt.xlabel('Степінь полінома (d)')
plt.ylabel('RMSE (менше — краще)')
plt.title('Крива валідації для поліноміальної складності')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Дослідження регуляризації на поліномі d=2
# Створення складності d=2 (77 ознак) для створення ризику перенавчання
alphas = np.logspace(-3, 3, 20)

reg_models = {
    'Ridge': lambda a: Ridge(alpha=a),
    'Lasso': lambda a: Lasso(alpha=a, max_iter=200000, tol=1e-4),
    'ElasticNet': lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=200000, tol=1e-4)
}

best_reg_results = []

plt.figure(figsize=(9, 4))

for name, model_func in reg_models.items():
    train_scores_list = []
    cv_scores_list = []
    
    for a in alphas:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', model_func(a))
        ])
        
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, 
            scoring='neg_root_mean_squared_error', 
            return_train_score=True
        )
        
        train_scores_list.append(-scores['train_score'].mean())
        cv_scores_list.append(-scores['test_score'].mean())
    
    # Визначення найкращого alpha
    best_idx = np.argmin(cv_scores_list)
    best_alpha = alphas[best_idx]
    best_cv_rmse = cv_scores_list[best_idx]
    best_train_rmse = train_scores_list[best_idx]
    
    best_reg_results.append({
        'Модель': name,
        'Найкраще alpha': round(best_alpha, 4),
        'Train RMSE': round(best_train_rmse, 4),
        'CV RMSE': round(best_cv_rmse, 4),
        'Gap': round(best_cv_rmse - best_train_rmse, 4)
    })
    
    plt.plot(alphas, cv_scores_list, label=f'{name} CV RMSE')

plt.xscale('log')
plt.xlabel('Сила регуляризації (alpha)')
plt.ylabel('CV RMSE')
plt.title('Вплив сили регуляризації (alpha) на валідаційну похибку')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Результати вибору найкращих регуляризованих моделей (d=2) ---")
print(pd.DataFrame(best_reg_results).to_string(index=False))

# 4. Фінальне оцінювання обраної моделі на тестовій вибірці
# Для порівняння обрано Ridge(alpha=0.001) на поліномі d=2
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=0.001))
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
r2_test = r2_score(y_test, y_pred)

print("\n--- Фінальна оцінка обраної регуляризованої моделі на тестовій вибірці ---")
print(f"Середня абсолютна похибка (MAE): {round(mae_test, 4)}")
print(f"Квадратична похибка (RMSE):       {round(rmse_test, 4)}")
print(f"Коефіцієнт детермінації (R2):    {round(r2_test, 4)}")