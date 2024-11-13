# AIT TAYEB LYES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger le dataset
data = pd.read_csv("heart.csv",sep=',')


print("Aperçu des premières lignes du dataset :")
print(data.head())

# Vérification de la taille et des types de données
print("\nInformations sur le dataset :")
print(data.info())

print("\nStatistiques descriptives :")
print(data.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(data.isnull().sum())



#  Préparation des données
X = data.drop(columns=['target'])
y = data['target']

# Division des données en ensembles d'entraînement et de test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle d'arbre de décision
depth = 3
clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
clf.fit(X, y)

# 4. Évaluation du modèle
# Prédiction sur l'ensemble de test
y_pred = clf.predict(X)
print("\nPrécision du modèle : {:.2f}%".format(accuracy_score(y, y_pred) * 100))
print("\nRapport de classification :")
print(classification_report(y, y_pred))

# Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'], filled=True)
plt.title("Arbre de décision pour prédiction de la maladie cardiaque")
plt.show()

# 5. Exploration des données et visualisations
# Histogramme de l'âge en fonction du diagnostic
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='age', hue='target', multiple='stack', palette="viridis")
plt.title("Distribution de l'âge par diagnostic de maladie cardiaque")
plt.xlabel("Âge")
plt.ylabel("Nombre de patients")
plt.show()

# Distribution du cholestérol par diagnostic
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='target', y='chol', palette="viridis")
plt.title("Distribution du cholestérol selon le diagnostic")
plt.xlabel("Diagnostic de maladie cardiaque (0=Non, 1=Oui)")
plt.ylabel("Niveau de cholestérol")
plt.show()

# Relation entre l'âge et la pression sanguine selon le diagnostic
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, x='age', y='trestbps', hue='target', palette="viridis")
plt.title("Relation entre l'âge et la pression sanguine en fonction du diagnostic")
plt.xlabel("Âge")
plt.ylabel("Pression sanguine au repos (trestbps)")
plt.show()

# Exploration des données et visualisations
# Scatter plot pour les classes 0 (absence de maladie) et 1 (présence de maladie)
# Utilisation de 2 caractéristiques arbitraires ici : 'age' et 'chol'

# Classe 0 (absence de maladie cardiaque)
abcisse_0 = X[y == 0]['age']
ordonnes_0 = X[y == 0]['chol']
plt.scatter(abcisse_0, ordonnes_0, label='No Heart Disease', color='blue')

# Classe 1 (présence de maladie cardiaque)
abcisse_1 = X[y == 1]['age']
ordonnes_1 = X[y == 1]['chol']
plt.scatter(abcisse_1, ordonnes_1, label='Heart Disease', color='red')


plt.xlabel('Âge')
plt.ylabel('Cholestérol')
plt.title('Répartition de l\'âge et du cholestérol en fonction de la présence de maladie cardiaque')
plt.legend()
plt.show()

# Scatter plot pour une autre paire de caractéristiques : 'trestbps' (pression sanguine au repos) et 'thalach' (fréquence cardiaque maximale)
# Classe 0
abcisse_0 = X[y == 0]['trestbps']
ordonnes_0 = X[y == 0]['thalach']
plt.scatter(abcisse_0, ordonnes_0, label='No Heart Disease', color='blue')

# Classe 1
abcisse_1 = X[y == 1]['trestbps']
ordonnes_1 = X[y == 1]['thalach']
plt.scatter(abcisse_1, ordonnes_1, label='Heart Disease', color='red')

plt.xlabel('Pression sanguine au repos (trestbps)')
plt.ylabel('Fréquence cardiaque maximale (thalach)')
plt.title('Relation entre pression sanguine au repos et fréquence cardiaque maximale')
plt.legend()
plt.show()
