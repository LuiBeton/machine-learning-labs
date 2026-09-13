import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('processed_tertyshnyk.csv')

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Обсяг навчальної вибірки: {X_train.shape[0]}")
print(f"Обсяг тестової вибірки: {X_test.shape[0]}")

models = {
    "Базове рішення": DummyRegressor(strategy="mean"),
    "Лінійна регресія": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    "Рідж регресія": Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ]),
    "Дерево рішень": DecisionTreeRegressor(max_depth=5, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -scores
    
    results.append({
        "Модель": name,
        "Середній RMSE": rmse_scores.mean(),
        "Розкид (Std)": rmse_scores.std()
    })

results_df = pd.DataFrame(results)
print("\n--- Результати 5-fold крос-валідації ---")
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(
    results_df["Модель"], 
    results_df["Середній RMSE"], 
    yerr=results_df["Розкид (Std)"], 
    capsize=5, 
    color=['gray', 'steelblue', 'darkblue', 'indianred']
)
plt.ylabel("RMSE (менше — краще)")
plt.title("Порівняння регресійних моделей за 5-fold крос-валідацією")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
r2_test = r2_score(y_test, y_pred)

print("\n--- Фінальна оцінка обраної моделі на тестовій вибірці ---")
print(f"Середня абсолютна похибка (MAE): {round(mae_test, 4)}")
print(f"Квадратична похибка (RMSE):       {round(rmse_test, 4)}")
print(f"Коефіцієнт детермінації (R2):    {round(r2_test, 4)}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='maroon')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Фактичні значення якості")
plt.ylabel("Передбачені значення якості")
plt.title("Фактичні та передбачені значення на тестовій вибірці")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()