import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

# Configuration et dépendances
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style="darkgrid")

# Initialisation des variables globales
uploaded_file = None
data = None
categorical_columns = []
numeric_columns = []
models = {}
performance = {}

# Disposition et Navigation de l'Application
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Aller à", ["Téléchargement de Fichier", "EDA", "Modélisation", "Prédiction"])

# Téléchargement de Fichier
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = data.select_dtypes(include=['float', 'int']).columns.tolist()

if page == "Téléchargement de Fichier":
    st.title("Téléchargement de Fichier")
    if uploaded_file is not None:
        with st.spinner('Chargement du fichier...'):
            st.success('Fichier chargé avec succès!')
            st.write("Aperçu du fichier téléchargé:")
            st.write(data.head())
    else:
        st.warning("Veuillez télécharger un fichier CSV.")

# Onglet d'Analyse Exploratoire des Données (EDA)
elif page == "EDA":
    st.title("Analyse Exploratoire des Données (EDA)")
    if uploaded_file is not None:
        with st.spinner('Traitement des données...'):
            # Statistiques de base
            st.header("Statistiques de base")
            st.write("Statistiques des données numériques:")
            st.write(data.describe())

            st.write("Statistiques des données catégorielles:")
            st.write(data.describe(include=['object']))

            # Visualisation des distributions
            st.header("Visualisation des distributions")
            for column in numeric_columns:
                st.write(f"Distribution de {column}:")
                fig, ax = plt.subplots()
                sns.histplot(data[column], kde=True, ax=ax)
                st.pyplot(fig)

                st.write(f"Boxplot de {column}:")
                fig, ax = plt.subplots()
                sns.boxplot(x=data[column], ax=ax)
                st.pyplot(fig)

            # Visualisation des relations
            st.header("Visualisation des relations")
            st.write("Matrice de corrélation:")
            fig, ax = plt.subplots()
            sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write("Scatter plot des paires de variables numériques:")
            pairplot = sns.pairplot(data[numeric_columns])
            st.pyplot(pairplot)

            # Gestion des valeurs manquantes
            st.header("Gestion des valeurs manquantes")
            missing_values = data.isnull().sum()
            st.write(missing_values[missing_values > 0])

            st.write("Pourcentage de valeurs manquantes:")
            st.write((missing_values[missing_values > 0] / len(data)) * 100)

            if st.button('Interpoler les valeurs manquantes'):
                data = data.interpolate()
                st.success('Valeurs manquantes interpolées!')
                st.write(data.isnull().sum())

            # Visualisation des données catégorielles
            st.header("Visualisation des données catégorielles")
            for column in categorical_columns:
                st.write(f"Distribution de {column}:")
                fig, ax = plt.subplots()
                sns.countplot(y=data[column], ax=ax)
                st.pyplot(fig)
    else:
        st.warning("Veuillez d'abord télécharger un fichier CSV.")

# Onglet d'Apprentissage Automatique (ML)
elif page == "Modélisation":
    st.title("Modélisation")
    if uploaded_file is not None:
        task = st.selectbox("Sélectionnez la tâche", ["Classification", "Régression"])
        target = st.selectbox("Sélectionnez la variable cible", data.columns)
        features = data.drop(columns=[target])
        
        # Encoder les variables catégorielles
        categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            encoder = OneHotEncoder()
            encoded_features = encoder.fit_transform(features[categorical_columns]).toarray()
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
            features = features.drop(columns=categorical_columns)
            features = pd.concat([features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(features, data[target], test_size=0.2, random_state=42)
        
        if task == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }
        
        st.write("Entraînement des modèles...")
        performance = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task == "Classification":
                performance[name] = {
                    "Précision": accuracy_score(y_test, y_pred),
                    "Rappel": recall_score(y_test, y_pred, average='weighted'),
                    "Précision": precision_score(y_test, y_pred, average='weighted'),
                    "F1-score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                performance[name] = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "R^2": r2_score(y_test, y_pred)
                }
        
        st.write("Métriques de performance:")
        st.write(pd.DataFrame(performance))
    else:
        st.warning("Veuillez d'abord télécharger un fichier CSV.")

# Interface de Prédiction
elif page == "Prédiction":
    st.title("Prédiction")
    if uploaded_file is not None and models:
        model_name = st.selectbox("Sélectionnez un modèle entraîné", list(models.keys()))
        if model_name:
            model = models.get(model_name)
            if model:
                new_data_input = st.text_input("Saisissez les nouvelles données (sous forme de dictionnaire)")
                if new_data_input:
                    try:
                        new_data = eval(new_data_input)
                        new_data_df = pd.DataFrame([new_data])

                        # Encoder les nouvelles données si nécessaire
                        if categorical_columns:
                            encoded_new_data = encoder.transform(new_data_df[categorical_columns]).toarray()
                            encoded_new_data_df = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out(categorical_columns))
                            new_data_df = new_data_df.drop(columns=categorical_columns)
                            new_data_df = pd.concat([new_data_df.reset_index(drop=True), encoded_new_data_df.reset_index(drop=True)], axis=1)

                        prediction = model.predict(new_data_df)
                        st.write("Prédiction:", prediction)
                    except Exception as e:
                        st.error(f"Erreur lors de l'évaluation des nouvelles données : {e}")
            else:
                st.error("Modèle non trouvé.")
        else:
            st.error("Aucun modèle sélectionné.")
    else:
        st.warning("Veuillez d'abord charger un fichier et entraîner les modèles.")
