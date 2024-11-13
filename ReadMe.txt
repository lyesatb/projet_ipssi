AIT TAYEB LYES 

##### petites explications #######

Ce projet utilise un arbre de décision pour prédire la présence ou l'absence de maladie cardiaque chez les patients. Le modèle est entraîné sur un ensemble de données (heart.csv) contenant plusieurs caractéristiques des patients, comme leur âge, leur cholestérol, leur pression sanguine, etc.



L'objectif de ce projet est de prédire si un patient présente des signes de maladie cardiaque à partir de ses caractéristiques cliniques. On utilise un arbre de décision pour classer les patients en deux catégories :

0 : Pas de maladie cardiaque
1 : Présence de maladie cardiaque


################################      Structure du Code   ######################################

1. Chargement et Exploration des Données
Le fichier heart.csv est chargé et exploré pour comprendre sa structure et vérifier s'il manque des données.
Nous affichons les premières lignes du dataset avec data.head(), et nous vérifions s'il y a des valeurs manquantes avec data.isnull().sum().

2. Préparation des Données
Nous séparons les données en deux parties :

X : les caractéristiques des patients (comme l'âge, le cholestérol, etc.).
y : la variable cible (présence ou absence de la maladie cardiaque).


3. Création du Modèle
Un arbre de décision est créé pour prédire si un patient a une maladie cardiaque ou non. On utilise la bibliothèque DecisionTreeClassifier de Scikit-learn et on définit une profondeur maximale de 3.

4. Évaluation du Modèle
Après avoir entraîné le modèle, nous évaluons sa performance en calculant sa précision sur l'ensemble des données. La précision est le pourcentage de prédictions correctes. Nous affichons également un rapport de classification avec des métriques comme la précision, le rappel, et la F-mesure.


5. Visualisation de l'Arbre de Décision
L'arbre de décision est visualisé pour mieux comprendre comment le modèle prend ses décisions. Chaque nœud montre la question posée sur une caractéristique, et les feuilles indiquent la classe prédite (maladie cardiaque ou non).


6. Visualisation des Données
Nous créons également quelques graphes pour mieux comprendre la distribution des caractéristiques et leur relation avec la présence de la maladie cardiaque. Par exemple, nous visualisons la distribution de l'âge et du cholestérol en fonction du diagnostic avec des graphiques scatter et boxplot.