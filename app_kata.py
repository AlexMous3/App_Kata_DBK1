from datetime import timedelta

from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import bartlett
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from scipy.stats import kendalltau, spearmanr
from scipy.stats import kruskal
from scipy.stats import f_oneway
import prince  # Pour l'ACM
import plotly.graph_objects as go
import streamlit as st

#print(prince.__version__)


# %%
#@st.cache_data
@st.cache_data
def load_data():
    data = pd.read_csv('Database_K1_2024.csv', sep=';', encoding='utf-8')

    # Remplacer les virgules par des points dans la colonne 'Note'
    data['Note'] = data['Note'].astype(str).str.replace(',', '.', regex=False)

    # Conversion des variables numériques
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce').astype('Int64')
    data['Ranking'] = pd.to_numeric(data['Ranking'], errors='coerce').astype('Int64')
    data['Note'] = pd.to_numeric(data['Note'], errors='coerce')

    # Conversion des autres variables en catégorielles
    variables_numeriques = ['Age', 'Ranking', 'Note']
    variables_categorielles = [col for col in data.columns if col not in variables_numeriques]
    
    for col in variables_categorielles:
        data[col] = data[col].astype('string').astype('category')

    return data

data = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Générateur de graphiques intéractifs", "ACM", "Focus Athlète"])

with tab1:
    st.header("Affichage du Dataset de Karaté")

    colonnes = data.columns.tolist()
    colonnes_selectionnees = st.multiselect("Sélectionnez les colonnes à afficher", colonnes, default=colonnes)

    if 'Grade' in data.columns:
        grades = data['Grade'].unique()
        grade_selection = st.multiselect("Filtrer par Grade", grades)
        if grade_selection:
            data = data[data['Grade'].isin(grade_selection)]

    st.dataframe(data[colonnes_selectionnees])

    @st.cache_data
    def convertir_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convertir_csv(data[colonnes_selectionnees])

    st.download_button(
        label="Télécharger le CSV filtré",
        data=csv,
        file_name='dataset_karate_filtre.csv',
        mime='text/csv',
    )


with tab2:
    st.header("Générateur de Graphiques Interactifs")

    st.markdown("""
    ### Comment utiliser les graphiques interactifs :

    1. **Sélectionnez la variable que vous souhaitez étudier en Y**, en fonction d'une autre variable en X. Par exemple, si vous voulez voir la **victoire** en fonction de la **ceinture**, choisissez **"Victoire"** en variable catégorielle Y, et **"Ceinture"** en variable catégorielle X.
    2. **Une seule variable par axe** : Vous pouvez choisir une variable pour l'axe X et une variable pour l'axe Y.
    3. **Effectuer un test statistique** : Après avoir affiché le graphique correspondant, vous pouvez cliquer sur **"Effectuer le test statistique"** pour étudier la relation entre les variables sélectionnées.
    4. **Distribution de variable numérique** : Pour étudier la distribution d'une variable numérique (exemple la note), il suffit de mettre la **variable numérique en Y**, sans mettre aucune variable en X
    """)

    # Séparer les variables numériques et catégorielles
    variables_numeriques = ['Age', 'Ranking', 'Note']
    variables_categorielles = [col for col in data.columns if col not in variables_numeriques]

    st.subheader("Sélection des variables pour les axes")

    # Axe X
    st.markdown("**Axe X**")
    x_num = st.selectbox("Variable numérique pour l'axe X", ["Aucune"] + variables_numeriques)
    x_cat = st.selectbox("Variable catégorielle pour l'axe X", ["Aucune"] + variables_categorielles)

    # Axe Y
    st.markdown("**Axe Y**")
    y_num = st.selectbox("Variable numérique pour l'axe Y", ["Aucune"] + variables_numeriques)
    y_cat = st.selectbox("Variable catégorielle pour l'axe Y", ["Aucune"] + variables_categorielles)

    st.subheader("Type de graphique généré")

    # Cas 1 : Une seule variable numérique en Y
    if y_num != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
        st.markdown(f"**Distribution de la variable {y_num}**")

        # Option pour filtrer sur une variable catégorielle
        filtre_var = st.selectbox("Sélectionnez une variable catégorielle pour filtrer (optionnel)", ["Aucune"] + variables_categorielles)
        if filtre_var != "Aucune":
            modalites = data[filtre_var].dropna().unique().tolist()
            modalites_selectionnees = st.multiselect(f"Filtrer les modalités de {filtre_var}", modalites, default=modalites)
            data_filtered = data[data[filtre_var].isin(modalites_selectionnees)]
        else:
            data_filtered = data

        data_filtered = data_filtered[data_filtered[y_num].notna()]

        # Graphique interactif avec Plotly
        fig = px.histogram(data_filtered, x=y_num, nbins=30, marginal='box')
        mean_value = data_filtered[y_num].mean()
        median_value = data_filtered[y_num].median()
        fig.add_vline(x=mean_value, line_dash='dash', line_color='red', annotation_text='Moyenne', annotation_position='top left')
        fig.add_vline(x=median_value, line_dash='dash', line_color='green', annotation_text='Médiane', annotation_position='top right')
        st.plotly_chart(fig)

        # Section "Test Statistique"
        st.subheader("Test Statistique")
        if st.button("Effectuer le test de normalité (Shapiro-Wilk)"):
            stat, p_value = stats.shapiro(data_filtered[y_num].dropna())
            if p_value > 0.05:
                interpretation = f"La distribution de **{y_num}** semble suivre une loi normale (p = {p_value:.3f})."
            else:
                interpretation = f"La distribution de **{y_num}** ne suit pas une loi normale (p = {p_value:.3f})."

            st.write(interpretation)
            st.markdown("""
            Le test de Shapiro-Wilk vérifie si les données suivent une distribution normale. 
            - **Hypothèse nulle (H₀)** : Les données suivent une distribution normale.
            - **Résultat** : Si la valeur p est supérieure à 0,05, on ne rejette pas H₀ et on considère que les données sont normales.
            """)

    # Cas 2 : Variable numérique en Y et variable catégorielle en X
    elif y_num != "Aucune" and x_cat != "Aucune" and x_num == "Aucune" and y_cat == "Aucune":
        st.markdown(f"**Distribution de {y_num} par rapport à chaque modalité de {x_cat}**")

        # Option pour filtrer sur une troisième variable catégorielle
        filtre_var = st.selectbox("Sélectionnez une variable catégorielle pour filtrer (optionnel)", ["Aucune"] + variables_categorielles)
        if filtre_var != "Aucune":
            modalites = data[filtre_var].dropna().unique().tolist()
            modalites_selectionnees = st.multiselect(f"Filtrer les modalités de {filtre_var}", modalites, default=modalites)
            data_filtered = data[data[filtre_var].isin(modalites_selectionnees)]
        else:
            data_filtered = data

        data_filtered = data_filtered[data_filtered[y_num].notna()]

        # Afficher le boxplot interactif avec Plotly
        fig = px.box(data_filtered, x=x_cat, y=y_num, points='all')
        st.plotly_chart(fig)

        # Section "Test Statistique"
        st.subheader("Test Statistique")
        if st.button("Effectuer le test statistique"):
            # Vérifier la normalité de la variable Y dans chaque groupe
            groups = [group[y_num].dropna() for name, group in data_filtered.groupby(x_cat)]
            normal = all(stats.shapiro(group)[1] > 0.05 for group in groups if len(group) >= 3)

            if normal:
                # ANOVA
                stat, p_value = stats.f_oneway(*groups)
                test_name = "ANOVA"
                interpretation = f"Le test ANOVA a été effectué."
            else:
                # Kruskal-Wallis
                stat, p_value = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"
                interpretation = f"Le test de Kruskal-Wallis a été effectué."

            if p_value < 0.05:
                conclusion = f"Il y a une différence significative de **{y_num}** entre les groupes de **{x_cat}** (p = {p_value:.3f})."
            else:
                conclusion = f"Aucune différence significative de **{y_num}** entre les groupes de **{x_cat}** n'a été trouvée (p = {p_value:.3f})."

            st.write(interpretation)
            st.write(conclusion)
            st.markdown(f"""
            - **Hypothèse nulle (H₀)** : Il n'y a pas de différence de **{y_num}** entre les groupes de **{x_cat}**.
            - **Résultat** : Si la valeur p est inférieure à 0,05, on rejette H₀, ce qui suggère une différence significative entre les groupes.
            """)

    # Cas 3 : Une seule variable catégorielle en Y
    elif y_cat != "Aucune" and x_num == "Aucune" and x_cat == "Aucune" and y_num == "Aucune":
        # Option pour filtrer sur une variable catégorielle
        filtre_var = st.selectbox("Sélectionnez une variable catégorielle pour filtrer (optionnel)", ["Aucune"] + variables_categorielles)

        if filtre_var != "Aucune":
            modalites = data[filtre_var].dropna().unique().tolist()
            modalites_selectionnees = st.multiselect(f"Filtrer les modalités de {filtre_var}", modalites, default=modalites)
            data_filtered = data[data[filtre_var].isin(modalites_selectionnees)]
        else:
            data_filtered = data

        # Histogramme des effectifs
        st.markdown(f"**Histogramme des effectifs de chaque modalité de {y_cat}**")
        counts = data_filtered[y_cat].value_counts().reset_index()
        counts.columns = [y_cat, 'Effectif']
        fig_count = px.bar(counts, x=y_cat, y='Effectif')
        st.plotly_chart(fig_count)

        # Histogramme des proportions
        st.markdown(f"**Histogramme des proportions de chaque modalité de {y_cat}**")
        counts_prop = data_filtered[y_cat].value_counts(normalize=True).reset_index()
        counts_prop.columns = [y_cat, 'Proportion']
        fig_prop = px.bar(counts_prop, x=y_cat, y='Proportion')
        st.plotly_chart(fig_prop)

    # Cas 4 : Variables catégorielles en X et en Y
    elif y_cat != "Aucune" and x_cat != "Aucune" and y_num == "Aucune" and x_num == "Aucune":
        st.markdown(f"**Proportions des modalités de {y_cat} en fonction de {x_cat}**")

        # Option pour filtrer sur une variable catégorielle
        filtre_var = st.selectbox("Sélectionnez une variable catégorielle pour filtrer (optionnel)", ["Aucune"] + variables_categorielles)

        if filtre_var != "Aucune":
            modalites = data[filtre_var].dropna().unique().tolist()
            modalites_selectionnees = st.multiselect(f"Filtrer les modalités de {filtre_var}", modalites, default=modalites)
            data_filtered = data[data[filtre_var].isin(modalites_selectionnees)]
        else:
            data_filtered = data

        # Calculer le tableau croisé des proportions
        crosstab = pd.crosstab(data_filtered[x_cat], data_filtered[y_cat], normalize='index')
        crosstab.reset_index(inplace=True)
        crosstab_melted = crosstab.melt(id_vars=x_cat, var_name=y_cat, value_name='Proportion')

        # Générer le graphique
        fig = px.bar(crosstab_melted, x=x_cat, y='Proportion', color=y_cat, barmode='stack')
        st.plotly_chart(fig)

        # Section "Test Statistique"
        st.subheader("Test Statistique")
        if st.button("Effectuer le test du Chi-deux"):
            contingency_table = pd.crosstab(data_filtered[x_cat], data_filtered[y_cat])
            stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            if p_value < 0.05:
                conclusion = f"Il existe une association significative entre **{x_cat}** et **{y_cat}** (p = {p_value:.3f})."
            else:
                conclusion = f"Aucune association significative entre **{x_cat}** et **{y_cat}** n'a été trouvée (p = {p_value:.3f})."

            st.write(conclusion)
            st.markdown(f"""
            - **Hypothèse nulle (H₀)** : Il n'y a pas d'association entre **{x_cat}** et **{y_cat}**.
            - **Résultat** : Si la valeur p est inférieure à 0,05, on rejette H₀, ce qui suggère une association significative entre les variables.
            """)


    # Cas 5 : Variables numériques en X et Y
    elif x_num != "Aucune" and y_num != "Aucune" and x_cat == "Aucune" and y_cat == "Aucune":
        st.markdown(f"**Nuage de points entre {x_num} et {y_num}**")

        # Option pour filtrer sur une variable catégorielle
        filtre_var = st.selectbox("Sélectionnez une variable catégorielle pour filtrer (optionnel)", ["Aucune"] + variables_categorielles)
        if filtre_var != "Aucune":
            modalites = data[filtre_var].dropna().unique().tolist()
            modalites_selectionnees = st.multiselect(f"Filtrer les modalités de {filtre_var}", modalites, default=modalites)
            data_filtered = data[data[filtre_var].isin(modalites_selectionnees)]
        else:
            data_filtered = data

        data_filtered = data_filtered[data_filtered[x_num].notna() & data_filtered[y_num].notna()]

        # Graphique interactif avec Plotly
        fig = px.scatter(data_filtered, x=x_num, y=y_num)
        st.plotly_chart(fig)

        # Section "Test Statistique"
        st.subheader("Test Statistique")
        if st.button("Effectuer le test de corrélation"):
            # Vérifier la normalité des variables
            normal_x = stats.shapiro(data_filtered[x_num])[1] > 0.05
            normal_y = stats.shapiro(data_filtered[y_num])[1] > 0.05

            if normal_x and normal_y:
                # Test de corrélation de Pearson
                corr_coeff, p_value = stats.pearsonr(data_filtered[x_num], data_filtered[y_num])
                test_name = "Pearson"
            else:
                # Test de corrélation de Spearman
                corr_coeff, p_value = stats.spearmanr(data_filtered[x_num], data_filtered[y_num])
                test_name = "Spearman"

            if p_value < 0.05:
                conclusion = f"Il existe une corrélation significative entre **{x_num}** et **{y_num}** (p = {p_value:.3f}, coefficient = {corr_coeff:.2f})."
            else:
                conclusion = f"Aucune corrélation significative entre **{x_num}** et **{y_num}** n'a été trouvée (p = {p_value:.3f})."

            st.write(f"Le test de corrélation de {test_name} a été effectué.")
            st.write(conclusion)
            st.markdown(f"""
            - **Hypothèse nulle (H₀)** : Il n'y a pas de corrélation entre **{x_num}** et **{y_num}**.
            - **Résultat** : Si la valeur p est inférieure à 0,05, on rejette H₀, ce qui suggère une corrélation significative entre les variables.
            - **Coefficient de corrélation** : Indique la force et le sens de la relation entre les variables.
            """)

    else:
        st.warning("Veuillez sélectionner des variables appropriées pour générer un graphique.")



with tab3:
    st.header("Analyse des Correspondances Multiples (ACM)")

    # Créer une copie des données pour éviter de modifier le DataFrame original
    data_acm = data.copy()

    # Créer la variable "Type de Tour" sans la modalité 'Autre'
    def create_type_de_tour(row):
        if row['N_Tour'] in ['Bronze', 'Final']:
            return 'Match de médaille'
        elif row['N_Tour'] in ['R1', 'R2']:
            return 'Quart (R1), Demi (R2)'
        elif row['N_Tour'] in ['Pool_1']:
            return 'Tour1'
        elif row['N_Tour'] in ['Pool_2']:
            return 'Tour2'
        elif row['N_Tour'] in ['Pool_3']:
            return 'Tour3'
        else:
            return None  # Retourne None au lieu de 'Autre'

    # Appliquer la fonction pour créer "Type de Tour"
    data_acm['Type de Tour'] = data_acm.apply(create_type_de_tour, axis=1)

    # Supprimer les lignes où "Type de Tour" est None
    data_acm = data_acm.dropna(subset=['Type de Tour'])

    # Filtrage sur "Type de Tour"
    type_de_tour_modalities = data_acm['Type de Tour'].unique().tolist()
    selected_types = st.multiselect(
        "Sélectionnez le(s) Type(s) de Tour à inclure dans l'ACM",
        type_de_tour_modalities,
        default=type_de_tour_modalities
    )

    if selected_types:
        data_acm_filtered = data_acm[data_acm['Type de Tour'].isin(selected_types)]
    else:
        st.warning("Aucun 'Type de Tour' sélectionné. Veuillez en sélectionner au moins un.")
        st.stop()

    # Filtrage sur la variable "Sexe"
    sexe_modalities = ['Aucun', 'M', 'F']
    selected_sexe = st.selectbox(
        "Sélectionnez le sexe à inclure dans l'ACM",
        sexe_modalities,
        index=0  # 'Aucun' est l'option par défaut
    )

    if selected_sexe != 'Aucun':
        data_acm_filtered = data_acm_filtered[data_acm_filtered['Sexe'] == selected_sexe]
        if data_acm_filtered.empty:
            st.warning("Aucune donnée disponible pour le sexe sélectionné. Veuillez choisir une autre option.")
            st.stop()

    # Option pour retirer des modalités de "Kata"
    kata_modalities = data_acm_filtered['Kata'].dropna().unique().tolist()
    selected_katas = st.multiselect(
        "Sélectionnez les modalités de 'Kata' à inclure dans l'ACM",
        kata_modalities,
        default=kata_modalities
    )

    if selected_katas:
        data_acm_filtered = data_acm_filtered[data_acm_filtered['Kata'].isin(selected_katas)]
        if data_acm_filtered.empty:
            st.warning("Aucune donnée disponible pour les modalités de 'Kata' sélectionnées. Veuillez choisir d'autres options.")
            st.stop()
    else:
        st.warning("Aucune modalité de 'Kata' sélectionnée. Veuillez en sélectionner au moins une.")
        st.stop()

    # Préparer les données pour l'ACM
    mca_variables = ['Kata', 'N_Tour', 'Victoire']
    data_mca = data_acm_filtered[mca_variables]

    # Supprimer les lignes avec des valeurs manquantes
    data_mca = data_mca.dropna()

    # Vérifier si le DataFrame est vide après suppression des valeurs manquantes
    if data_mca.empty:
        st.warning("Les données sont vides après suppression des valeurs manquantes. Impossible de réaliser l'ACM.")
        st.stop()

    # Réinitialiser les index
    data_mca.reset_index(drop=True, inplace=True)

    # S'assurer que les variables sont de type catégoriel
    for col in mca_variables:
        data_mca[col] = data_mca[col].astype(str).astype('category')

    # Options d'affichage
    st.subheader("Options d'affichage de l'ACM")
    display_individuals = st.checkbox("Afficher les individus", value=False)
    display_modalities = st.checkbox("Afficher les modalités des variables", value=True)
    st.write("*Note : L'affichage des ellipses de confiance à 95% n'est pas actuellement pris en charge.*")

    # Effectuer l'ACM
    mca = prince.MCA(n_components=2, random_state=42)
    mca = mca.fit(data_mca)

    # Obtenir les coordonnées des modalités des variables
    modalities_coords = mca.column_coordinates(data_mca)
    modalities_coords['Variable'] = [var.split('_')[0] for var in modalities_coords.index]
    modalities_coords['Modalité'] = [var.split('_')[1] for var in modalities_coords.index]

    # Obtenir les coordonnées des individus
    if display_individuals:
        individuals_coords = mca.row_coordinates(data_mca)

    # Calculer l'inertie expliquée manuellement
    eigenvalues = mca.eigenvalues_
    total_inertia = eigenvalues.sum()
    explained_inertia = eigenvalues / total_inertia

    # Créer le graphique
    fig = go.Figure()

    # Tracer les modalités des variables
    if display_modalities:
        for variable in mca_variables:
            var_coords = modalities_coords[modalities_coords['Variable'] == variable]
            fig.add_trace(go.Scatter(
                x=var_coords[0],
                y=var_coords[1],
                mode='markers+text',
                name=f"Modalités de {variable}",
                text=var_coords['Modalité'],
                textposition='top center',
                marker=dict(size=10),
            ))

    # Tracer les individus
    if display_individuals:
        fig.add_trace(go.Scatter(
            x=individuals_coords[0],
            y=individuals_coords[1],
            mode='markers',
            name='Individus',
            marker=dict(size=5, color='grey', opacity=0.5),
            text=data_mca.index.astype(str),
            hoverinfo='text',
        ))

    # Configurer la mise en page
    fig.update_layout(
        title="Représentation de l'ACM",
        xaxis_title=f"Dimension 1 ({explained_inertia[0]*100:.2f}% d'inertie)",
        yaxis_title=f"Dimension 2 ({explained_inertia[1]*100:.2f}% d'inertie)",
        showlegend=True,
        width=800,
        height=600,
    )

    # Calculer les limites des axes
    if display_individuals:
        x_coords = pd.concat([modalities_coords[0], individuals_coords[0]])
        y_coords = pd.concat([modalities_coords[1], individuals_coords[1]])
    else:
        x_coords = modalities_coords[0]
        y_coords = modalities_coords[1]

    x_min = x_coords.min() - 0.5
    x_max = x_coords.max() + 0.5
    y_min = y_coords.min() - 0.5
    y_max = y_coords.max() + 0.5

    # Mettre à jour les limites des axes
    fig.update_xaxes(range=[x_min, x_max], zeroline=False)
    fig.update_yaxes(range=[y_min, y_max], zeroline=False)

    # Ajouter les axes à x=0 et y=0
    fig.add_shape(
        type="line",
        x0=x_min,
        y0=0,
        x1=x_max,
        y1=0,
        line=dict(color="black", width=1)
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=y_min,
        x1=0,
        y1=y_max,
        line=dict(color="black", width=1)
    )

    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=False)

    # Ajouter le texte explicatif
    st.subheader("Interprétation de l'ACM")

    st.markdown("""
    L'analyse des correspondances multiples (ACM) est une méthode statistique utilisée pour explorer les relations entre plusieurs variables qualitatives (catégorielles). Elle permet de représenter graphiquement les modalités des variables et les individus dans un espace de dimensions réduites.

    **Comment interpréter ce graphique :**

    - **Dimensions :** Les axes (Dimension 1 et Dimension 2) représentent les directions principales dans lesquelles les données varient.
    - **Modalités proches :** Les modalités qui sont proches sur le graphique sont souvent associées, c'est-à-dire que les individus qui choisissent une modalité ont tendance à choisir l'autre.
    - **Dimension 1 associée à la victoire :** Dans cette analyse, la Dimension 1 est fortement associée à la variable **"Victoire"**.
      - Les modalités et individus situés du côté de True de la Dimension 1 sont associés aux **victoires** .
      - Ceux du côté de False sont associés aux **défaites** .

    **Exemple d'interprétation :**

    - Si un **Kata** particulier est positionné du côté True de la Dimension 1, cela suggère que ce kata est souvent réalisé lors de victoires.
    - De même, les **modalités de "N_Tour"** situées près du côté True de la Dimension 1 sont associées aux tours où les victoires sont plus fréquentes.

    **Note :** Cette interprétation est une simplification et doit être confirmée par une analyse plus approfondie.

    N'hésitez pas à explorer les données en modifiant les filtres ci-dessus pour voir comment cela affecte le graphique et les relations entre les variables.
    """)

with tab4:
    st.header("Focus Athlète")

    # Obtenir la liste des athlètes uniques
    athlete_names = data['Nom'].dropna().unique().tolist()
    athlete_names.sort()  # Trier les noms pour une meilleure expérience utilisateur

    # Sélectionner un athlète
    selected_athlete = st.selectbox("Sélectionnez un athlète", athlete_names)

    # Filtrer les données pour l'athlète sélectionné
    athlete_data = data[data['Nom'] == selected_athlete]

    # Extraire les informations
    sexe = athlete_data['Sexe'].mode()[0] if not athlete_data['Sexe'].mode().empty else "Non spécifié"
    age_mean = athlete_data['Age'].mean()
    ranking_min = athlete_data['Ranking'].min()
    ranking_max = athlete_data['Ranking'].max()
    nation = athlete_data['Nation'].mode()[0] if not athlete_data['Nation'].mode().empty else "Non spécifié"
    style = athlete_data['Style'].mode()[0] if not athlete_data['Style'].mode().empty else "Non spécifié"

    # Formater le ranking
    if ranking_min == ranking_max or pd.isna(ranking_min) or pd.isna(ranking_max):
        ranking_str = f"{ranking_min}" if not pd.isna(ranking_min) else "Non spécifié"
    else:
        ranking_str = f"{ranking_min} - {ranking_max}"

    # Disposition en colonnes
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Informations de l'athlète")
        st.markdown(f"""
        - **Nom :** {selected_athlete}
        - **Sexe :** {sexe}
        - **Âge :** {age_mean:.1f} ans
        - **Ranking :** {ranking_str}
        - **Nationalité :** {nation}
        - **Style :** {style}
        """)

    with col2:
        # Dictionnaire pour les noms des compétitions
        competition_names = {
            'FRA': 'K1 Paris',
            'TUR': 'K1 Antalya',
            'EGY': 'K1 Cairo',
            'MAR': 'K1 Casablanca'
        }

        # Ajouter une colonne avec le nom de la compétition
        athlete_data['Compétition'] = athlete_data['Pays_compet'].map(competition_names)

        # Dictionnaire pour les niveaux de tours
        tour_levels = {
            'Pool_1': 1,
            'Pool_2': 1,
            'Pool_3': 1,
            'R1': 2,
            'R2': 3,
            'Bronze': 4,
            'Final': 5
        }

        tour_names = {
            1: 'Poule',
            2: 'Quart de finale',
            3: 'Demi finale',
            4: 'Place de 3',
            5: 'Final'
        }

        # Ajouter une colonne avec le niveau numérique du tour
        athlete_data['Niveau_Tour'] = athlete_data['N_Tour'].map(tour_levels)

        # Calculer le tour maximal atteint par compétition
        max_tours = athlete_data.groupby('Compétition')['Niveau_Tour'].max().reset_index()
        max_tours['Tour_Max'] = max_tours['Niveau_Tour'].map(tour_names)

        st.subheader("Tour maximal atteint par compétition")
        # Créer le graphique
        fig = px.bar(
            max_tours,
            x='Compétition',
            y='Niveau_Tour',
            text='Tour_Max',
            labels={'Niveau_Tour': 'Tour maximal atteint'},
            range_y=[0, 5.5]  # Pour inclure tous les niveaux
        )

        # Personnaliser les axes
        fig.update_yaxes(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Poule', 'Quart de finale', 'Demi finale', 'Place de 3', 'Final']
        )

        # Ajouter les labels sur les barres
        fig.update_traces(textposition='outside')

        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)

    # Section 3 : Histogramme des Katas effectués
    st.subheader("3. Histogramme des Katas effectués")

    # Sélectionner les tours pour le filtre
    tour_options = athlete_data['N_Tour'].dropna().unique().tolist()
    selected_tours = st.multiselect("Filtrer par tour (N_Tour)", options=tour_options, default=tour_options)

    # Filtrer les données en fonction des tours sélectionnés
    kata_data = athlete_data[athlete_data['N_Tour'].isin(selected_tours)]

    # Compter les Katas
    kata_counts = kata_data['Kata'].value_counts().reset_index()
    kata_counts.columns = ['Kata', 'Nombre']

    # Ne conserver que les Katas avec un count > 0
    kata_counts = kata_counts[kata_counts['Nombre'] > 0]

    # Vérifier qu'il y a des Katas à afficher
    if kata_counts.empty:
        st.warning("Aucun Kata à afficher pour les tours sélectionnés.")
    else:
        # Créer l'histogramme
        fig_kata = px.bar(
            kata_counts,
            x='Kata',
            y='Nombre',
            title='Nombre de Katas effectués',
            labels={'Nombre': 'Nombre de fois'},
            text='Nombre'
        )
        fig_kata.update_layout(xaxis_title='Kata', yaxis_title='Nombre de fois')
        fig_kata.update_traces(textposition='outside')
        st.plotly_chart(fig_kata, use_container_width=True)

    # Section 4 : Diagramme de Kiviat des notes par N_Tour
    st.subheader("4. Diagramme de Kiviat de la moyenne des notes par N_Tour")
    # Sélectionner les compétitions pour le filtre
    competition_options = athlete_data['Pays_compet'].dropna().unique().tolist()
    selected_competitions = st.multiselect("Filtrer par compétition", options=competition_options, default=competition_options)

    # Filtrer les données en fonction des compétitions sélectionnées
    note_data = athlete_data[athlete_data['Pays_compet'].isin(selected_competitions)]

    # Calculer la moyenne des notes par N_Tour
    n_tour_levels = ['Pool_1', 'Pool_2', 'Pool_3', 'R1', 'R2', 'Bronze', 'Final']
    average_notes = []

    for tour in n_tour_levels:
        tour_data = note_data[note_data['N_Tour'] == tour]
        # Exclure les valeurs de 'Note' nulles ou NaN
        tour_data = tour_data.dropna(subset=['Note'])
        if not tour_data.empty:
            avg_note = tour_data['Note'].mean()
        else:
            avg_note = None
        average_notes.append(avg_note)

    # Préparer les données pour le diagramme de Kiviat
    df_kiviat_tour = pd.DataFrame({
        'N_Tour': n_tour_levels,
        'Moyenne_Note': average_notes
    })


    # Supprimer les tours sans moyenne de note
    df_kiviat_tour = df_kiviat_tour.dropna(subset=['Moyenne_Note'])


    # Vérifier qu'il y a des données à afficher
    if df_kiviat_tour.empty:
        st.warning("Aucune donnée de note disponible pour les tours sélectionnés.")
    else:
        # Définir le range radial
        min_note = 35.0
        max_note = 50.0

        # Créer le diagramme de Kiviat
        fig_kiviat_tour = go.Figure()

        fig_kiviat_tour.add_trace(go.Scatterpolar(
            r=df_kiviat_tour['Moyenne_Note'],
            theta=df_kiviat_tour['N_Tour'],
            fill='toself',
            name='Moyenne des Notes'
        ))

        # Mise en page du graphique
        fig_kiviat_tour.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_note, max_note]
                )
            ),
            showlegend=False,
            title="Moyenne des notes par N_Tour"
        )

    # Section 5 : Diagramme de Kiviat des notes par Kata
    st.subheader("5. Diagramme de Kiviat de la moyenne des notes par Kata")
    # Calculer la moyenne des notes par Kata
    kata_list = athlete_data['Kata'].dropna().unique().tolist()
    average_notes_kata = []

    for kata in kata_list:
        kata_data = athlete_data[athlete_data['Kata'] == kata]
        # Exclure les valeurs de 'Note' nulles ou NaN
        kata_data = kata_data.dropna(subset=['Note'])
        if not kata_data.empty:
            avg_note = kata_data['Note'].mean()
        else:
            avg_note = None
        average_notes_kata.append(avg_note)

    # Préparer les données pour le diagramme de Kiviat
    df_kiviat_kata = pd.DataFrame({
        'Kata': kata_list,
        'Moyenne_Note': average_notes_kata
    })

    # Supprimer les Katas sans moyenne de note
    df_kiviat_kata = df_kiviat_kata.dropna(subset=['Moyenne_Note'])

    # Vérifier qu'il y a des données à afficher
    if df_kiviat_kata.empty:
        st.warning("Aucune donnée de note disponible pour les Katas.")
    else:
        # Définir le range radial
        min_note_kata = 35.0
        max_note_kata = 50.0

        # Créer le diagramme de Kiviat
        fig_kiviat_kata = go.Figure()

        fig_kiviat_kata.add_trace(go.Scatterpolar(
            r=df_kiviat_kata['Moyenne_Note'],
            theta=df_kiviat_kata['Kata'],
            fill='toself',
            name='Moyenne des Notes'
        ))

        # Mise en page du graphique
        fig_kiviat_kata.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_note_kata, max_note_kata]
                )
            ),
            showlegend=False,
            title="Moyenne des notes par Kata"
        )

    # Disposition en colonnes pour les diagrammes de Kiviat
    col_kiviat1, col_kiviat2 = st.columns(2)

    with col_kiviat1:
        if not df_kiviat_tour.empty:
            st.plotly_chart(fig_kiviat_tour, use_container_width=True)

    with col_kiviat2:
        if not df_kiviat_kata.empty:
            st.plotly_chart(fig_kiviat_kata, use_container_width=True)


# Fonction pour ajouter le pied de page
def add_footer():
    footer = """
    <style>
    /* Positionnement du pied de page */
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        text-align: right;
        width: 100%;
        padding: 10px;
        font-size: 12px;
        color: grey;
    }
    /* Style pour la source en gris très clair */
    .footer .source {
        color: lightgrey;
        opacity: 0.7;
    }
    /* Style pour le texte sans marges */
    .footer p {
        margin: 0;
    }
    </style>
    <div class="footer">
        <p class="source">Source : SportData</p>
        <p>&copy; Alexis Vincent</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

# Appel de la fonction pour ajouter le pied de page
add_footer()
