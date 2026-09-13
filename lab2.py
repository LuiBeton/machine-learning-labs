import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)

data = fetch_openml(name='mushroom', version=1, as_frame=True, parser='auto')
X = data.data
y = (data.target == 'p').astype(int)

class_counts = y.value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(['Їстівні (0)', 'Отруйні (1)'], class_counts.values, color=['green', 'red'])
plt.ylabel("Кількість екземплярів")
plt.title("Розподіл класів у датасеті грибів")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Обсяг навчальної вибірки: {X_train.shape[0]}")
print(f"Обсяг тестової вибірки: {X_test.shape[0]}")

models = {
    "Базове рішення": DummyClassifier(strategy="most_frequent"),
    "Логістична регресія": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Наївний Байєс": Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', GaussianNB())
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    cv_results.append({
        "Модель": name,
        "Accuracy": round(scores['test_accuracy'].mean(), 4),
        "Precision": round(scores['test_precision'].mean(), 4),
        "Recall": round(scores['test_recall'].mean(), 4),
        "F1": round(scores['test_f1'].mean(), 4),
        "ROC-AUC": round(scores['test_roc_auc'].mean(), 4)
    })

cv_df = pd.DataFrame(cv_results)
print("\n--- Результати 5-fold крос-валідації ---")
print(cv_df.to_string(index=False))

best_model = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Їстівні (0)', 'Отруйні (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Матриця помилок на тестовій вибірці")
plt.tight_layout()
plt.show()

print("\n--- Фінальні метрики на тестовій вибірці ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

errors = X_test.copy()
errors["Фактичний"] = y_test
errors["Передбачений"] = y_pred
errors["Ймовірність"] = y_proba
errors["Тип помилки"] = np.where(
    (errors["Фактичний"] == 0) & (errors["Передбачений"] == 1), "FP",
    np.where((errors["Фактичний"] == 1) & (errors["Передбачений"] == 0), "FN", "ОК")
)

misclassified = errors[errors["Тип помилки"] != "ОК"].copy()
misclassified["Відстань до порога"] = (misclassified["Ймовірність"] - 0.5).abs()

print(f"\nЗагальна кількість помилок на тестовій вибірці: {len(misclassified)}")
if len(misclassified) > 0:
    print(misclassified[["Фактичний", "Передбачений", "Ймовірність", "Тип помилки", "Відстань до порога"]])

thresholds = [0.3, 0.5, 0.7]
thresh_results = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    thresh_results.append({
        "Поріг (t)": t,
        "Precision": round(precision_score(y_test, y_pred_t), 4),
        "Recall": round(recall_score(y_test, y_pred_t), 4),
        "F1": round(f1_score(y_test, y_pred_t), 4),
        "FP (Хибна тривога)": cm_t[0, 1],
        "FN (Пропущена отрута)": cm_t[1, 0]
    })

thresh_df = pd.DataFrame(thresh_results)
print("\n--- Вплив порогу класифікації на метрики ---")
print(thresh_df.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.plot(thresh_df["Поріг (t)"], thresh_df["Precision"], marker='o', label="Precision")
plt.plot(thresh_df["Поріг (t)"], thresh_df["Recall"], marker='s', label="Recall")
plt.plot(thresh_df["Поріг (t)"], thresh_df["F1"], marker='^', label="F1-score")
plt.xlabel("Поріг класифікації")
plt.ylabel("Значення метрики")
plt.title("Залежність метрик від порогу прийняття рішення")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-крива (AUC = {auc_val:.4f})", color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("Частка хибних спрацьовувань (FPR)")
plt.ylabel("Частка правильних виявлень (TPR)")
plt.title("ROC-крива для моделі Логістичної регресії")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()