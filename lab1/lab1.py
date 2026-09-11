import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('processed_tertyshnyk.csv')

print("1. Перевірка розмірності та перших рядків")
print(f"Розмір таблиці: {df.shape}")
print(df.head())


print("\n2. Кількість вин за оцінками якості")
counts = df['quality'].value_counts().sort_index()
print(counts)


good_wines = (df['quality'] >= 6).mean() * 100
print(f"\n3. Відсоток хороших вин (>=6): {round(good_wines, 2)}%")


mean_alko = df['alcohol'].mean()
median_alko = df['alcohol'].median()
print(f"\n4. Середній алкоголь: {round(mean_alko, 2)}, медіана: {round(median_alko, 2)}")


citric_corr = df['volatile acidity'].corr(df['citric acid'])
print(f"\n5. Кореляція між volatile acidity та citric acid: {round(citric_corr, 3)}")


plt.figure(figsize=(7, 4))
df['alcohol'].plot(kind='hist', bins=12, color='maroon', edgecolor='black')
plt.title('Розподіл вмісту алкоголю')
plt.xlabel('Алкоголь')
plt.ylabel('Кількість')
plt.grid(True, alpha=0.3)
plt.show()


print("\n7. Топ оцінок якості")
print(df['quality'].value_counts().head())


plt.figure(figsize=(7, 4))
df['quality'].value_counts().sort_index().plot(kind='bar', color='firebrick')
plt.title('Кількість вин за якістю')
plt.xlabel('Оцінка якості')
plt.ylabel('Кількість зразків')
plt.grid(axis='y', alpha=0.3)
plt.show()


print("\n9. Середні значення за групами якості")
grouped_data = df.groupby('quality')[['alcohol', 'pH']].mean()
print(grouped_data)