import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Завантаження та огляд
dani = pd.read_csv('winequality-red.csv')

print("1. Перші рядки датасету:")
print(dani.head())
print("\nРозмірність таблиці:", dani.shape)
print("Кількість пропусків:", dani.isnull().sum().sum())

# 2. Обробка пропусків
for k in dani.columns:
    if dani[k].isnull().sum() > 0:
        dani[k] = dani[k].fillna(dani[k].median())

# 3. Видалення дублікатів
d = dani.duplicated().sum()
print("\n3. Знайдено дублікатів:", d)
if d > 0:
    dani = dani.drop_duplicates()
    print("Розмірність після видалення дублікатів:", dani.shape)

# 4. Графіки та обробка викидів
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=dani['alcohol'])
plt.title('Alcohol (до обмеження)')

granicya = dani['total sulfur dioxide'].quantile(0.99)
dani.loc[dani['total sulfur dioxide'] > granicya, 'total sulfur dioxide'] = granicya

plt.subplot(1, 2, 2)
sns.boxplot(x=dani['total sulfur dioxide'])
plt.title('Total Sulfur Dioxide (після обмеження)')
plt.tight_layout()
plt.show()

# 5. Масштабування
sc = StandardScaler()
kolonky = [c for c in dani.columns if c != 'quality']

for c in kolonky:
    dani[c + '_scaled'] = sc.fit_transform(dani[[c]])

print("\n5. Масштабування виконано успішно.")

# 6. Кодування
# У цьому датасеті всі вхідні ознаки є числовими

# 7. Збереження результату
X_dani = dani.drop(columns=['quality'])
y_dani = dani['quality']

res_df = pd.concat([X_dani, y_dani], axis=1)
res_df.to_csv('processed_tertyshnyk.csv', index=False)
print("\n7. Готово! Файл збережено як 'processed_tertyshnyk.csv'")