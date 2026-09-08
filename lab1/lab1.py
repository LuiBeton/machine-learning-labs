import pandas as pd
import matplotlib.pyplot as plt

dani = pd.read_csv('titanic.csv')

# 1. choloviky i zhinky
m = len(dani[dani['Sex'] == 'male'])
z = len(dani[dani['Sex'] == 'female'])
print("1. Чоловіків і жінок:", m, z)

# 2. vidsotok vyzhyvshykh
vyzhyly = dani['Survived'].mean() * 100
print("2. Відсоток виживших:", round(vyzhyly, 2), "%")

# 3. vidsotok 1 klasu
klas1 = (dani['Pclass'] == 1).mean() * 100
print("3. Відсоток 1 класу:", round(klas1, 2), "%")

# 4. seredniy vik i mediana
s_vik = dani['Age'].mean()
m_vik = dani['Age'].median()
print("4. Середній вік і медіана:", round(s_vik, 2), round(m_vik, 2))

# 5. korelyatsiya
kor = dani['SibSp'].corr(dani['Parch'])
print("5. Кореляція:", round(kor, 4), "(помірна)")

# 6. gistograma (підписи українською)
dani['Age'].plot(kind='hist', bins=16)
plt.title('Розподіл віку пасажирів')
plt.xlabel('Вік')
plt.ylabel('Кількість')
plt.show()

# 7. populyarni imena
zhin_imena = dani[dani['Sex'] == 'female']['Name']
print("\n7. Популярні імена:")
print(zhin_imena.value_counts().head(5))

# 8. grafik po klasakh (перекладено назви для легенди та осі)
dani_grafik = dani.copy()
dani_grafik['Sex'] = dani_grafik['Sex'].map({'male': 'Чоловіки', 'female': 'Жінки'})
dani_grafik.groupby(['Pclass', 'Sex']).size().unstack().plot(kind='bar')
plt.title('Пасажири за класом і статтю')
plt.xlabel('Клас (Pclass)')
plt.ylabel('Кількість')
plt.show()

# 9. seredniy vik по групі (перекладено індекси статі)
s_grupy = dani.groupby(['Sex', 'Pclass'])['Age'].mean()
s_grupy.index = s_grupy.index.set_levels(['Жінки', 'Чоловіки'], level=0)
s_grupy.index.names = ['Стать', 'Клас']
print("\n9. Середній вік за статтю та класом:")
print(s_grupy)