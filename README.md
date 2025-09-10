# 📊 Prédiction du PIB Réel du Bénin avec un Modèle VAR

## 📝 Contexte

Ce projet vise à modéliser et prédire l'évolution du Produit Intérieur Brut (PIB) réel du Bénin à court terme (horizon de 5 ans).

L'approche repose sur un modèle économétrique VAR (Vector AutoRegressive) appliqué aux taux de croissance de trois variables clés :

* 📈 **PIB réel** (dollars constants de 2015)
* 🏗️ **Investissement** (Formation Brute de Capital Fixe)
* 🌍 **Balance commerciale**

L'utilisation du PIB réel permet de neutraliser les effets de l'inflation et de se concentrer sur la croissance économique tangible.

## 🎯 Objectifs

* Construire un modèle robuste pour prédire la croissance du PIB réel
* Mettre en évidence le rôle des investissements et du secteur extérieur dans la dynamique économique du Bénin
* Fournir une application interactive de visualisation et d'exploration via Streamlit

## 🏗️ Architecture du projet

### 1. Entraînement et sérialisation du modèle (`train_and_serialize_model.py`)

* Chargement et préparation des données (source : Banque mondiale)
* Transformation en taux de croissance pour assurer la stationnarité
* Estimation d'un modèle VAR avec sélection optimale du lag via le critère AIC
* Diagnostic des résidus :
  * Durbin-Watson (autocorrélation)
  * Shapiro-Wilk (normalité)
  * Test de White (homoscédasticité)
* Sérialisation du modèle et des diagnostics dans un fichier `growth_model_bundle.pkl`

### 2. Application Streamlit (`app.py`)

* **Page Accueil** : présentation du projet, fondements théoriques, méthodologie
* **Analyse descriptive** :
  * Exploration des données macroéconomiques (PIB, investissement, balance commerciale)
  * Visualisations interactives
* **Analyse économétrique** :
  * Prédictions du PIB réel sur 5 ans
  * Intervalle de confiance à 95%
  * Visualisations des projections
  * Diagnostic du modèle

## ⚙️ Installation

### Prérequis

* Python 3.9+
* pip ou conda

### Dépendances

Installer les librairies nécessaires :

```bash
pip install -r requirements.txt
