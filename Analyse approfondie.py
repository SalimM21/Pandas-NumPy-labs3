# Charger le fichier employees2.csv dans un DataFrame Pandas
import pandas as pd
import numpy as np

df = pd.read_csv('employees2.csv')
# Afficher les 5 premières lignes du DataFrame
print(df.head( n=5))

# Vérifier les types de données de chaque colonne
print(df.dtypes)

# Identifier les valeurs manquantes par colonne
print(df.isnull().sum())

# Partie 2 : Nettoyage des données
# Remplacer les valeurs manquantes dans la colonne Age par la médiane de cette colonne
df['Age']=df['Age'].fillna(df['Age'].median())
print(df ['Age'])

# Remplir les valeurs manquantes dans Salaire en utilisant la moyenne par département 
df['salary']=df['Salary'].fillna(df.groupby('Department')['Salary'].transform('mean'))
print(df['Salary'])

# Convertir toutes les colonnes numériques en type approprié (float ou int)
for col in df.select_dtypes(include=['number']).columns:
    if (df[col] % 1 == 0).all():  # Si toutes les valeurs sont des entiers
            df[col] = df[col].astype(int)
    else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Afficher les types de données après conversion
print(df.dtypes)

# Remplacer les valeurs 'Yes'/'No' dans Remote par 'Oui'/'Non'
df['Remote'] = df['Remote'].replace({'Yes': 'Oui', 'No': 'Non'})
print(df['Remote'])

# Créer une nouvelle colonne Ancienneté_Catégorie qui classe les années d’expérience en :
# Junior : < 3 ans
# Intermédiaire : 3–7 ans
# Senior : 8–15 ans
# Expert : > 15 ans

def categorize_experience(years):
    if years < 3:
        return 'Junior'
    elif 3 <= years < 8:
        return 'Intermédiaire'
    elif 8 <= years <= 15:
        return 'Senior'
    else:
        return 'Expert'
df['Ancienneté_Catégorie'] = df['Years_Experience'].apply(categorize_experience)
print(df[['Years_Experience', 'Ancienneté_Catégorie']])

# Calculer le salaire moyen global
print(df['Salary'].mean())

# Trouver l’employé(e) avec le salaire le plus élevé
highest_salary_employee = df.loc[df['Salary'].idxmax()]
print("Employé(e) avec le salaire le plus élevé :") 
print(highest_salary_employee[['Name', 'Salary']])

# Calculer le salaire moyen par département
salary_by_department = df.groupby('Department')['Salary'].mean().reset_index()
print("\nSalaire moyen par département :")
print(salary_by_department)

# Calculer la moyenne et la médiane des salaires par groupe d’ancienneté
salary_by_experience = df.groupby('Ancienneté_Catégorie')['Salary'].agg(['mean', 'median']).reset_index()
print("\nMoyenne et médiane des salaires par groupe d’ancienneté :")
print(salary_by_experience)

# Compter combien d’employés travaillent en télétravail (Remote) par département
remote_count_by_department = df[df['Remote'] == 'Oui'].groupby('Department').size().reset_index(name='Count')
print("\nNombre d’employés en télétravail par département :")
print(remote_count_by_department)

# Créer un tableau croisé dynamique montrant le salaire moyen par département et par télétravail
pivot_table = df.pivot_table(values='Salary', index='Department', columns='Remote', aggfunc='mean').reset_index()
print("\nTableau croisé dynamique du salaire moyen par département et par télétravail :")
print(pivot_table)

# Créer un autre tableau croisé dynamique montrant le nombre moyen d’années d’expérience par groupe d’âge et par département
pivot_table_experience = df.pivot_table(values='Years_Experience', index='Age', columns='Department', aggfunc='mean').reset_index()
print("\nTableau croisé dynamique du nombre moyen d’années d’expérience par groupe d’âge et par département :")
print(pivot_table_experience)

# Utiliser np.where() pour créer une colonne Performance :
# "Bon" si Salaire < 60000
# "Moyen" si 60000 ≤ Salaire < 80000
# "Haut" si Salaire ≥ 80000 

df['Performance'] = np.where(df['Salary'] < 60000, 'Bon',
                              np.where(df['Salary'] < 80000, 'Moyen', 'Haut'))
print("\nColonne Performance :")
print(df[['Name', 'Salary', 'Performance']])

# Utiliser np.select() pour classer les employés selon leur âge et leur ancienneté :
# Jeune & Nouveau
# Jeune & Expérimenté
# Senior & Nouveau
# Senior & Expérimenté

conditions = [
    (df['Age'] < 30) & (df['Ancienneté_Catégorie'] == 'Junior'),
    (df['Age'] < 30) & (df['Ancienneté_Catégorie'] != 'Junior'),
    (df['Age'] >= 30) & (df['Ancienneté_Catégorie'] == 'Senior'),
    (df['Age'] >= 30) & (df['Ancienneté_Catégorie'] != 'Senior')
]
choices = [
    'Jeune & Nouveau',
    'Jeune & Expérimenté',
    'Senior & Nouveau',
    'Senior & Expérimenté'
]
df['Classification'] = np.select(conditions, choices, default='Autre')
print("\nClassification des employés :")
print(df[['Name', 'Age', 'Ancienneté_Catégorie', 'Classification']])

# Calculer la différence entre le salaire de chaque employé et le salaire moyen de son département
df['Salary_Difference'] = df['Salary'] - df.groupby('Department')['Salary'].transform('mean')
print("\nDifférence de salaire par rapport au salaire moyen du département :")
print(df[['Name', 'Department', 'Salary', 'Salary_Difference']])


# Utiliser Matplotlib ou Seaborn pour :
# Afficher la distribution des salaires
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Afficher la distribution des salaires
plt.figure(figsize=(10,6))
sns.histplot(df['Salary'], bins=10, kde=True)
plt.title('Distribution des salaires')
plt.show()

# Comparer les salaires moyens par département sous forme de barplot
plt.figure(figsize=(12,6))
sns.barplot(x='Department', y='Salary', data=salary_by_department, palette='viridis')
plt.title('Salaire moyen par département')
plt.xticks(rotation=45)
plt.show() 

# Boxplot des salaires par groupe d’ancienneté

plt.figure(figsize=(12,6))
sns.boxplot(x='Ancienneté_Catégorie', y='Salary', data=df, palette='Set2')
plt.title('Boxplot des salaires par groupe d’ancienneté')
plt.show()

