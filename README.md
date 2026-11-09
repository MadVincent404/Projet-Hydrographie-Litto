# HydroQC AI: démonstration Streamlit

Visualisation de résultats **pré-calculés** : plan robuste, Isolation Forest, LOF et PointNet 3D.
Aucun entraînement ni modèle PyTorch n'est exécuté par l'application.

## Contenu
- 5 dalles MNT5m réelles, une figure 3D par dalle : 0087_6834, 0088_6832, 0108_6838, 0087_6832, 0089_6838
- `scores_<dalle>.npz` : XYZ et scores calculés localement
- `leaderboard.csv` et `synthetic_per_tile.csv` : métriques sur anomalies injectées
- `run_summary.json` : métadonnées agrégées du benchmark

## Local
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Déployer
Crée un dépôt GitHub avec le contenu de ce dossier. Puis dans Streamlit Community Cloud, sélectionne le dépôt, la branche `main`, et le fichier principal `app.py`.

## Données et limites
Source des MNT : **Shom / Litto3D Finistère 2014** — https://diffusion.shom.fr/donnees/altimetrie-littorale/litto-3d/litto3dr-finistere-2014.html . Respecter les conditions de réutilisation du produit. **Ne pas utiliser pour la navigation.**

Les performances concernent des anomalies **synthétiques injectées** ; les sondages réels ne sont pas étiquetés « correct » ou « erreur ». Les anomalies affichées sont des *candidats à examiner*, pas des erreurs confirmées.
