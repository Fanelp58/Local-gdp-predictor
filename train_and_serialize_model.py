# ==============================================================================
# SCRIPT D'ENTRAÎNEMENT - STRATÉGIE 100% TAUX DE CROISSANCE AVEC DIAGNOSTICS
# ==============================================================================
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_white
from scipy.stats import shapiro
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

print("--- Début du processus d'entraînement (Stratégie Taux de Croissance) ---")

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
print("\n--- Étape 1: Chargement des Données ---")

try:
    df_full = pd.read_csv(
        "donnees_benin.csv", 
        sep=';',
        decimal=',',
        encoding='latin1',
        index_col="Année"
    )
    df_full.index = df_full.index.astype(int)
    print("✅ Données chargées.")
except Exception as e:
    print(f"❌ Erreur lors du chargement des données : {e}")
    exit()

df_full.columns = df_full.columns.str.strip()
for col in df_full.columns:
    if df_full[col].dtype == 'object':
        df_full[col] = pd.to_numeric(df_full[col].str.replace(',', '.'), errors='coerce')
        
df_system = df_full[['PIB', 'Investissement', 'Balance commerciale']].copy().dropna()

# ==============================================================================
# 2. CALCUL DES TAUX DE CROISSANCE
# ==============================================================================
print("\n--- Étape 2: Transformation de toutes les séries en taux de croissance ---")

df_growth = pd.DataFrame(index=df_system.index)
# Renommer les colonnes pour la clarté dans les résultats de diagnostic
df_growth['Croissance_PIB'] = df_system['PIB'].pct_change() * 100
df_growth['Croissance_Investissement'] = df_system['Investissement'].pct_change() * 100
df_growth['Croissance_Balance_Comm'] = df_system['Balance commerciale'].pct_change() * 100

df_growth.replace([np.inf, -np.inf], np.nan, inplace=True)
df_growth.dropna(inplace=True)

print("✅ Données transformées en taux de croissance.")

# ==============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE VAR
# ==============================================================================
print("\n--- Étape 3: Entraînement du Modèle VAR ---")

model = VAR(df_growth)
model_fit = model.fit(ic='aic', maxlags=3) 

print(f"✅ Modèle VAR entraîné avec succès (lag optimal p={model_fit.k_ar}).")

# ==============================================================================
# 4. DIAGNOSTIC DES RÉSIDUS (SECTION AJOUTÉE)
# ==============================================================================
print("\n--- Étape 4: Diagnostic des Résidus ---")
residuals = model_fit.resid

# Test de Durbin-Watson pour l'autocorrélation
dw_results = durbin_watson(residuals)
dw_values = {col: val for col, val in zip(df_growth.columns, dw_results)}
print("✅ Test de Durbin-Watson (Autocorrélation) effectué.")

# Test de Shapiro-Wilk pour la normalité
shapiro_p_values = {col: shapiro(residuals[col])[1] for col in residuals.columns}
print("✅ Test de Shapiro-Wilk (Normalité) effectué.")

# Test de White pour l'homoscédasticité
try:
    # Le test de White compare les résidus aux variables exogènes du modèle.
    # Dans un VAR, les exogènes sont les lags des variables elles-mêmes.
    exog_for_het_test = model_fit.model.exog
    white_test_p_value = het_white(residuals, exog_for_het_test)[1]
    print("✅ Test de White (Homoscédasticité) effectué.")
except Exception as e:
    white_test_p_value = None # Le test peut échouer si l'échantillon est trop petit
    print(f"⚠️ Le test de White n'a pas pu être effectué : {e}")

# Stocker les résultats des diagnostics dans un dictionnaire
diagnostics = {
    'durbin_watson': dw_values,
    'shapiro_wilk': shapiro_p_values,
    'white_test': white_test_p_value
}

# ==============================================================================
# 5. SÉRIALISATION DU MODÈLE ET DES DONNÉES
# ==============================================================================
print("\n--- Étape 5: Sérialisation des Artefacts ---")

bundle_for_app = {
    'model_fit': model_fit,
    'df_growth': df_growth,
    'df_full': df_full,
    'diagnostics': diagnostics  # <-- ON AJOUTE LES DIAGNOSTICS AU BUNDLE
}

bundle_filename = 'growth_model_bundle.pkl'
joblib.dump(bundle_for_app, bundle_filename)

print(f"✅ Tous les éléments ont été sérialisés dans le fichier : '{bundle_filename}'")
print("\n--- Processus terminé. ---")