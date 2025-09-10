import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Prédiction du PIB Réel du Bénin",
    page_icon="🇧🇯",
    layout="wide"
)

# CSS personnalisé avec couleurs variées
st.markdown("""
<style>
    /* Styles pour les titres principaux de page (sans zone de texte) */
    .page-title {
        color: #2E4057;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 1rem;
    }
    
    /* Zones de contenu avec différentes couleurs */
    .blue-box {
        background-color: #E8F4FD;
        color: #1A365D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4299E1;
    }
    
    .green-box {
        background-color: #F0FFF4;
        color: #22543D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #48BB78;
    }
    
    .orange-box {
        background-color: #FFFAF0;
        color: #7B341E;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ED8936;
    }
    
    .purple-box {
        background-color: #FAF5FF;
        color: #44337A;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #9F7AEA;
    }
    
    .teal-box {
        background-color: #E6FFFA;
        color: #234E52;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #38B2AC;
    }
    
    .pink-box {
        background-color: #FFF5F7;
        color: #702459;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ED64A6;
    }
    
    /* Titres de section centrés */
    .section-title {
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.8rem;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
        color: inherit;
    }
    
    /* Style pour les métriques */
    .metric-container {
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
    }
    
    .metric-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Styles pour les étapes de méthodologie */
    .methodology-step {
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .step-title {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.8rem;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
        color: inherit;
    }
    
    /* Dividers simples */
    .custom-divider {
        height: 2px;
        background-color: #CBD5E0;
        border: none;
        border-radius: 1px;
        margin: 2rem 0;
    }
    
    /* Style pour les cartes de variables */
    .variable-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
    }
    
    .variable-title {
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 5px;
        color: inherit;
    }
    
    /* Sidebar personnalisée */
    .sidebar-header {
        background-color: #F7FAFC;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    
    .sidebar-title {
        color: #2D3748;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FONCTIONS DE CHARGEMENT (mises en cache pour la performance)
# ==============================================================================

@st.cache_data
def load_raw_data(file_path):
    """Charge les données brutes depuis le fichier CSV pour l'analyse descriptive."""
    try:
        df = pd.read_csv(
            file_path, 
            sep=';',
            decimal=',',
            encoding='latin1',
            index_col="Année"
        )
        df.columns = df.columns.str.strip()
        # Conversion robuste en numérique
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
        return df.dropna()
    except FileNotFoundError:
        st.error(f"Fichier '{file_path}' introuvable. Assurez-vous qu'il est dans le même dossier que l'application.")
        return None

@st.cache_resource
def load_model_bundle(bundle_path):
    """Charge le 'bundle' contenant le modèle et les données nécessaires."""
    try:
        bundle = joblib.load(bundle_path)
        return bundle
    except FileNotFoundError:
        st.error(f"Fichier '{bundle_path}' introuvable. Veuillez exécuter le script 'train_and_serialize_model.py' d'abord.")
        return None

# ==============================================================================
# INTERFACE UTILISATEUR (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2 class="sidebar-title">Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Choisissez une page :", ["Accueil", "Analyse descriptive", "Analyse économétrique"])

# ==============================================================================
# PAGE 1 : ACCUEIL
# ==============================================================================
if page == "Accueil":
    st.markdown('<h1 class="page-title">Projet de prédiction du PIB Réel du Bénin</h1>', unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="blue-box">
        <div class="section-title">Contexte du projet</div>
        <p>Cette application web présente les résultats d'un projet de modélisation économétrique visant à prédire l'évolution du <strong>PIB Réel</strong> du Bénin sur un horizon de court terme de <strong>5 ans</strong>.</p>
        <p>L'utilisation du PIB Réel permet de se concentrer sur la croissance de la production en neutralisant les effets de l'inflation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="green-box">
        <div class="section-title">Objectif principal</div>
        <p>Construire un modèle robuste basé sur des données historiques pour fournir des prévisions quantitatives sur la croissance économique réelle du pays.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="orange-box">
        <div class="section-title">Fondements théoriques</div>
        <p>Le modèle est fondé sur la théorie macroéconomique. Il modélise le <strong>taux de croissance du PIB Réel</strong> en fonction de deux déterminants clés :</p>
        <ul>
            <li><strong>Le taux de croissance de l'investissement (FBCF)</strong> - moteur de la demande à court terme et de l'offre à long terme</li>
            <li><strong>Le taux de croissance de la balance commerciale</strong> - pour capturer l'influence dynamique du secteur extérieur</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="purple-box">
        <div class="section-title">Stratégie de modélisation</div>
        <p style="text-align: center;"><strong>Approche VAR sur les taux de croissance pour assurer la robustesse statistique</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blue-box methodology-step">
        <div class="step-title">1. Transformation des données en taux de croissance</div>
        <p>Les séries en niveau (PIB Réel, Investissement, Balance Commerciale) sont converties en taux de croissance annuels pour :</p>
        <ul>
            <li><strong>Assurer la stationnarité :</strong> Les taux de croissance sont naturellement stationnaires</li>
            <li><strong>Interprétation économique :</strong> Prédictions directement interprétables en points de croissance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="teal-box methodology-step">
        <div class="step-title">2. Validation de la stationnarité</div>
        <p>Test de <strong>Dickey-Fuller Augmenté (ADF)</strong> pour confirmer la stationnarité et éviter les régressions fallacieuses.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="green-box methodology-step">
        <div class="step-title">3. Justification du modèle VAR</div>
        <p>Après test de cointégration de Johansen (aucune relation détectée), le modèle VAR (Vector AutoRegressive) est optimal pour capturer les <strong>interdépendances dynamiques de court terme</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="orange-box methodology-step">
        <div class="step-title">4. Sélection de l'ordre optimal</div>
        <p>L'ordre du VAR est déterminé par minimisation du <strong>Critère d'Information d'Akaike (AIC)</strong> pour éviter le surajustement.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pink-box methodology-step">
        <div class="step-title">5. Prévision et reconstruction</div>
        <p><strong>Processus en deux étapes :</strong></p>
        <ol>
            <li>Prédiction des taux de croissance futurs via le modèle VAR</li>
            <li>Reconstruction des niveaux du PIB en dollars constants pour l'interprétation finale</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2 : ANALYSE DESCRIPTIVE
# ==============================================================================
elif page == "Analyse descriptive":
    st.markdown('<h1 class="page-title">Exploration des données</h1>', unsafe_allow_html=True)
    
    df = load_raw_data("donnees_benin.csv")

    if df is not None:
        st.markdown("""
        <div class="blue-box">
            <div class="section-title">Source et période des données</div>
            <p><strong>Source des données :</strong> Base de données de la <a href="https://databank.worldbank.org/source/world-development-indicators" target="_blank">Banque Mondiale</a></p>
            <p><strong>Période couverte :</strong> 1993 à 2024</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="purple-box">
            <div class="section-title">Clarification conceptuelle des variables</div>
            <p>Le choix des indicateurs est crucial pour la pertinence de l'analyse. Voici une description détaillée des variables utilisées :</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="green-box variable-card">
            <div class="variable-title">PIB Réel (Produit Intérieur Brut en dollars constants de 2015)</div>
            <p><strong>Définition :</strong> Le PIB est la mesure la plus large de l'activité économique, représentant la valeur totale de tous les biens et services finaux produits dans le pays sur une année.</p>
            <p><strong>Pourquoi le PIB <em>Réel</em> ?</strong> Nous utilisons le PIB "réel" plutôt que nominal pour neutraliser l'effet de l'inflation et mesurer la croissance réelle de la production.</p>
            <p><strong>Conclusion :</strong> Le PIB réel garantit que notre modèle se concentre sur la <strong>croissance réelle et tangible de l'économie</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="orange-box variable-card">
            <div class="variable-title">Investissement (Formation Brute de Capital Fixe, en dollars constants de 2015)</div>
            <p><strong>Définition :</strong> La FBCF inclut les investissements des entreprises (machines, usines) et du secteur public (infrastructures, écoles).</p>
            <p><strong>Pourquoi ce choix ?</strong> L'investissement est un moteur essentiel de la croissance future en augmentant le stock de capital productif. C'est un excellent <strong>indicateur avancé</strong> du PIB futur.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="teal-box variable-card">
            <div class="variable-title">Balance commerciale (en dollars courants)</div>
            <p><strong>Définition :</strong> Différence entre exportations et importations de biens et services.</p>
            <p><strong>Pourquoi ce choix ?</strong> Pour une économie ouverte comme le Bénin, le commerce extérieur est vital :</p>
            <ul>
                <li><strong>Excédent</strong> (exports > imports) : stimule la demande intérieure</li>
                <li><strong>Déficit</strong> : effet inverse</li>
            </ul>
            <p><strong>Note :</strong> Transformée en taux de croissance pour la stationnarité et l'analyse dynamique.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pink-box">
            <div class="section-title">Visualisation de l'évolution des variables</div>
        </div>
        """, unsafe_allow_html=True)
        
        options = st.multiselect(
            'Choisissez une ou plusieurs variables à visualiser :',
            df.columns.tolist(),
            default=['PIB', 'Investissement']
        )

        if options:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['#2E8B57', '#4682B4', '#FF6347', '#9370DB', '#20B2AA']
            
            for i, var in enumerate(options):
                ax.plot(df.index, df[var], label=var, marker='o', markersize=4, 
                       color=colors[i % len(colors)], linewidth=2.5)
            
            ax.set_title("Évolution des agrégats macroéconomiques au Bénin de 1993 à 2024", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Année", fontsize=12, fontweight='bold')
            ax.set_ylabel("Valeur", fontsize=12, fontweight='bold')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Amélioration esthétique
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            st.pyplot(fig)
        
        st.markdown("""
        <div class="blue-box">
            <div class="section-title">Table de données</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df.style.format("{:,.2f}"))

# ==============================================================================
# PAGE 3 : ANALYSE ÉCONOMÉTRIQUE
# ==============================================================================
elif page == "Analyse économétrique":
    st.markdown('<h1 class="page-title">Prédictions du modèle VAR</h1>', unsafe_allow_html=True)

    bundle = load_model_bundle('growth_model_bundle.pkl')

    if bundle:
        model_fit = bundle['model_fit']
        df_growth = bundle['df_growth']
        df_full = bundle['df_full']
        diagnostics = bundle['diagnostics']
        
        n_forecast = 5
        
        # Section de prédiction
        last_lags = df_growth.values[-model_fit.k_ar:]
        forecast_result = model_fit.forecast_interval(last_lags, steps=n_forecast, alpha=0.05)
        point_forecasts_growth, lower_ci_growth, upper_ci_growth = forecast_result
        cols = df_growth.columns
        forecast_growth_df = pd.DataFrame(point_forecasts_growth, columns=cols)
        lower_growth_df = pd.DataFrame(lower_ci_growth, columns=cols)
        upper_growth_df = pd.DataFrame(upper_ci_growth, columns=cols)
        
        def reconstruct_level_from_growth(growth_preds, last_level_value):
            reconstructed = []
            current_level = last_level_value
            for growth_rate in growth_preds:
                current_level *= (1 + growth_rate / 100)
                reconstructed.append(current_level)
            return reconstructed
            
        last_pib_level = df_full['PIB'].iloc[-1]
        last_year = df_full.index[-1]
        future_index = pd.RangeIndex(start=last_year + 1, stop=last_year + 1 + n_forecast)
        final_preds = pd.DataFrame(index=future_index, columns=['PIB prédit', 'PIB_Lower_CI', 'PIB_Upper_CI'])
        final_preds.index.name = 'Annee'
        final_preds['PIB prédit'] = reconstruct_level_from_growth(forecast_growth_df['Croissance_PIB'], last_pib_level)
        final_preds['PIB_Lower_CI'] = reconstruct_level_from_growth(lower_growth_df['Croissance_PIB'], last_pib_level)
        final_preds['PIB_Upper_CI'] = reconstruct_level_from_growth(upper_growth_df['Croissance_PIB'], last_pib_level)
        
        st.markdown("""
        <div class="blue-box">
            <div class="section-title">Graphique des prédictions du PIB Réel</div>
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(df_full.index, df_full['PIB'], label='PIB Réel historique (Niveau)', 
                color='#2E8B57', marker='o', markersize=5, linewidth=3)
        ax.plot(final_preds.index, final_preds['PIB prédit'], label='PIB Réel prédit (Niveau)', 
                color='#4682B4', linestyle='--', marker='x', markersize=8, linewidth=3)
        ax.fill_between(final_preds.index, final_preds['PIB_Lower_CI'], final_preds['PIB_Upper_CI'], 
                        color='#87CEEB', alpha=0.4, label='Intervalle de confiance à 95%')
        
        ax.set_title(f'Prédiction du PIB Réel du Bénin sur {n_forecast} ans', fontsize=20, fontweight='bold', pad=25)
        ax.set_ylabel('PIB (en dollars constants de 2015)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Année', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e9:.1f} Mds'))
        
        # Amélioration esthétique
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        st.pyplot(fig)

        st.markdown("""
        <div class="green-box">
            <div class="section-title">Analyse de l'évolution</div>
            <ul>
                <li><strong>Tendance historique :</strong> Le graphique illustre la croissance soutenue du PIB réel du Bénin, reflétant une augmentation de la production de biens et services au fil des décennies.</li>
                <li><strong>Tendance prédite :</strong> Le modèle VAR projette une continuation de cette dynamique de croissance sur les {n_forecast} prochaines années. L'intervalle de confiance s'élargit logiquement avec l'horizon de prévision.</li>
            </ul>
        </div>
        """.format(n_forecast=n_forecast), unsafe_allow_html=True)
                
        st.warning("""
        **Limites et précautions :**
        - Le modèle VAR est intrinsèquement un outil de **prévision de court terme**. Les prévisions au-delà de 2 à 3 ans doivent être interprétées avec prudence.
        - Ce projet est un **exercice de modélisation** basé sur les tendances passées et ne peut anticiper des chocs externes imprévus.
        """)
        
        st.markdown("""
        <div class="orange-box">
            <div class="section-title">Table des prédictions du PIB Réel sur {n_forecast} ans</div>
        </div>
        """.format(n_forecast=n_forecast), unsafe_allow_html=True)
        
        st.dataframe(final_preds[['PIB prédit', 'PIB_Lower_CI', 'PIB_Upper_CI']].style.format("{:,.0f} $"))

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="purple-box">
            <div class="section-title">Performance et diagnostic du modèle</div>
            <p>Validation de la fiabilité du modèle VAR par l'analyse des résidus. Les résidus doivent se comporter comme un "bruit blanc" : aléatoire, sans structure et imprévisible.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="blue-box metric-container">
                <div class="metric-title">Test d'autocorrélation</div>
                <p><strong>Hypothèse :</strong> Les résidus ne doivent pas être corrélés entre eux.</p>
            </div>
            """, unsafe_allow_html=True)
            
            dw_pib = diagnostics['durbin_watson']['Croissance_PIB']
            st.metric(label="Statistique de Durbin-Watson", value=f"{dw_pib:.2f}")
            
            if 1.5 < dw_pib < 2.5:
                st.success("Le test est concluant (valeur proche de 2), indiquant une absence d'autocorrélation significative.")
            else:
                st.warning("Le test suggère une possible autocorrélation dans les résidus.")
        
        with col2:
            st.markdown("""
            <div class="teal-box metric-container">
                <div class="metric-title">Test de normalité</div>
                <p><strong>Hypothèse :</strong> Les résidus doivent suivre une distribution normale.</p>
            </div>
            """, unsafe_allow_html=True)
            
            shapiro_pib = diagnostics['shapiro_wilk']['Croissance_PIB']
            st.metric(label="P-value du test de Shapiro-Wilk", value=f"{shapiro_pib:.3f}")
            
            if shapiro_pib > 0.05:
                st.success("Le test est concluant (p-value > 0.05), l'hypothèse de normalité n'est pas rejetée.")
            else:
                st.error("Le test a échoué (p-value < 0.05), les résidus ne sont probablement pas distribués normalement.")

        with col3:
            st.markdown("""
            <div class="pink-box metric-container">
                <div class="metric-title">Test d'homoscédasticité</div>
                <p><strong>Hypothèse :</strong> La variance des résidus doit être constante.</p>
            </div>
            """, unsafe_allow_html=True)
            
            white_p = diagnostics['white_test']
            if white_p is not None:
                st.metric(label="P-value du test de White", value=f"{white_p:.3f}")
                if white_p > 0.05:
                    st.success("Le test est concluant (p-value > 0.05), l'hypothèse d'homoscédasticité est validée.")
                else:
                    st.error("Le test a échoué (p-value < 0.05), les résidus sont hétéroscédastiques.")
            else:
                st.info("Le test de White n'a pas pu être calculé en raison de la petite taille de l'échantillon.")