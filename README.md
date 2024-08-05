# ia-1

Description du Projet
Ce projet utilise Streamlit pour créer une application interactive de traitement et d'analyse de données. L'application permet aux utilisateurs de télécharger des fichiers CSV, d'explorer les données à l'aide d'analyses exploratoires, de créer et d'évaluer des modèles d'apprentissage automatique, et de faire des prédictions avec des modèles entraînés.

Fonctionnalités :

Téléchargement de Fichiers :

Les utilisateurs peuvent télécharger un fichier CSV qui est ensuite chargé dans l'application. Les colonnes du fichier sont automatiquement détectées comme catégorielles ou numériques.
Analyse Exploratoire des Données (EDA) :

Statistiques de Base : Affichage des statistiques descriptives pour les données numériques et catégorielles.
Visualisations : Création de graphiques pour les distributions des variables numériques (histogrammes et boxplots), matrices de corrélation, et visualisation des relations entre variables.
Gestion des Valeurs Manquantes : Affichage des valeurs manquantes et possibilité d'interpolation pour combler les lacunes.
Visualisation des Catégories : Affichage des distributions des variables catégorielles à l'aide de diagrammes en barres.
Modélisation :

Sélection de la Tâche : Choix entre classification et régression.
Prétraitement des Données : Encodage des variables catégorielles et séparation des données en ensembles d'entraînement et de test.
Entraînement des Modèles : Entraînement de différents modèles de machine learning (régression logistique, régression linéaire, forêt aléatoire pour la classification et la régression).
Évaluation des Modèles : Calcul et affichage des métriques de performance, telles que la précision, le rappel, la précision (pour la classification) et l'erreur quadratique moyenne, le R² (pour la régression).
Prédiction :

Sélection du Modèle : Les utilisateurs peuvent choisir un modèle entraîné pour faire des prédictions.
Entrée des Nouvelles Données : Les utilisateurs peuvent entrer de nouvelles données sous forme de dictionnaire pour obtenir des prédictions à partir du modèle sélectionné.
Encodage des Nouvelles Données : Les nouvelles données sont encodées de la même manière que les données d'entraînement avant la prédiction.
Technologies Utilisées :

Streamlit pour l'interface utilisateur interactive.
Pandas pour la manipulation des données.
Matplotlib et Seaborn pour la visualisation des données.
Scikit-learn pour l'apprentissage automatique et l'évaluation des modèles.
Ce projet est conçu pour fournir une interface conviviale pour le traitement de données, l'exploration et la modélisation, tout en permettant des ajustements interactifs et une visualisation facile des résultats.
