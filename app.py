"""Tableau de bord HydroQC AI : comparaison ML / DL sur des dalles Litto3D.

L'application présente des résultats déjà calculés : aucun modèle n'est entraîné
ou exécuté dans Streamlit. Elle permet de :

- comparer les performances sur des anomalies artificiellement injectées ;
- localiser les candidats dans le plan XY ;
- visualiser chaque dalle sur son propre relief MNT5m en 3D ;
- exporter les points suspects en CSV.

Lancement, depuis le dossier contenant ce fichier :
    python -m streamlit run app_final.py

IMPORTANT : une détection est un point à examiner, pas une erreur confirmée.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from affine import Affine
import streamlit as st

# 1. CONFIGURATION GÉNÉRALE
# Les chemins relatifs sont résolus depuis ce fichier, et non depuis le terminal.
# Le dashboard fonctionne donc aussi après publication sur Streamlit Cloud.
DOSSIER_APPLICATION = Path(__file__).resolve().parent

METHODES = ("plane", "iforest", "lof", "pointnet")

NOMS_METHODES = {
    "plane": "Plan local robuste",
    "iforest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "pointnet": "PointNet 3D",
}

COULEURS_METHODES = {
    "plane": "#ef8b2c",
    "iforest": "#23a097",
    "lof": "#9f72d0",
    "pointnet": "#e25166",
}

FICHIERS_BENCHMARK = (
    "leaderboard.csv",
    "synthetic_per_tile.csv",
    "run_summary.json",
)


@dataclass
class OptionsAffichage:
    """Regroupe les choix de l'utilisateur dans la barre latérale."""

    dalles: list[str]
    methodes: list[str]
    fraction_alertes: float
    ecart_vertical_min: float
    points_2d: int
    points_3d: int
    exageration_verticale: int
    manifeste: Path
    pas_raster: int
    afficher_sondages: bool
    afficher_courbes: bool
    palette: str


def chemin_application(chemin: str | Path) -> Path:
    """Convertit un chemin relatif en chemin situé dans le dépôt GitHub."""
    resultat = Path(chemin).expanduser()
    if resultat.is_absolute():
        return resultat
    return DOSSIER_APPLICATION / resultat

# 2. CHARGEMENT DES RÉSULTATS (AVEC CACHE STREAMLIT)
@st.cache_data(show_spinner=False)
def charger_scores_xyz(chemin: str) -> dict[str, np.ndarray]:
    """Lit un NPZ pré-calculé : points XYZ, masques et scores des modèles."""
    with np.load(chemin, allow_pickle=False) as fichier:
        return {nom: fichier[nom] for nom in fichier.files}


@st.cache_data(show_spinner=False)
def charger_csv(chemin: str) -> pd.DataFrame:
    """Évite de relire les CSV à chaque changement de filtre."""
    return pd.read_csv(chemin)


@st.cache_data(show_spinner=False)
def charger_mnt(chemin: str, pas: int = 1):
    """Lit le véritable MNT5m de Litto3D et son géoréférencement.

    ``pas=2`` conserve un pixel sur deux pour accélérer l'affichage.
    Les positions correspondent au centre des pixels du raster source.
    Aucune surface n'est interpolée à partir des sondages XYZ.

    Renvoie : x, y, altitudes, transformation affine, CRS.
    """
    if pas not in (1, 2, 4):
        raise ValueError("Le pas d'échantillonnage doit être 1, 2 ou 4.")

    with rasterio.open(chemin) as raster:
        altitudes = raster.read(1, masked=True).astype(np.float32).filled(np.nan)
        altitudes[altitudes <= -9000] = np.nan  # Autre code NoData possible.

        grille_z = altitudes[::pas, ::pas]
        transformation = raster.transform

        indices_colonnes = np.arange(0, raster.width, pas)
        indices_lignes = np.arange(0, raster.height, pas)

        x = transformation.c + (indices_colonnes + 0.5) * transformation.a
        y = transformation.f + (indices_lignes + 0.5) * transformation.e

        coefficients = tuple(transformation)[:6]
        systeme_coordonnees = raster.crs.to_string() if raster.crs else None

    return x, y, grille_z, coefficients, systeme_coordonnees


def charger_chemins_mnt(manifeste: Path) -> dict[str, str]:
    """Associe chaque identifiant de dalle au fichier MNT du manifeste."""
    if not manifeste.is_file():
        return {}

    contenu = json.loads(manifeste.read_text(encoding="utf-8"))
    chemins = {}

    # Plusieurs formats de manifestes existent selon les versions du pipeline.
    for partition in ("train", "val", "test", "val_tune", "val_calibration"):
        for dalle in contenu.get(partition, []):
            if "tile_id" in dalle and "mnt_path" in dalle:
                chemins[dalle["tile_id"]] = dalle["mnt_path"]

    return chemins


def trouver_mnt(dalle_id: str, chemins_mnt: dict, resume: dict) -> Path | None:
    """Localise le MNT5m correspondant à la dalle affichée.

    Recherche dans l'ordre : manifeste, dépôt public, ancienne arborescence XYZ.
    Si aucun fichier réel n'est trouvé, l'application montre les points XYZ.
    """
    chemin_connu = chemins_mnt.get(dalle_id)
    if chemin_connu:
        fichier = chemin_application(chemin_connu)
        if fichier.is_file():
            return fichier

    # L'export public place les seuls rasters retenus dans data/mnt5m/.
    fichier_public = DOSSIER_APPLICATION / "data" / "mnt5m" / f"{dalle_id}.asc"
    if fichier_public.is_file():
        return fichier_public

    # Compatibilité avec les anciens résultats qui mémorisaient le chemin XYZ.
    for entree in resume.get("raw_display_tiles", []):
        if entree.get("tile_id") != dalle_id:
            continue

        chemin_xyz = chemin_application(entree.get("xyz_path", ""))
        dossier_mnt = chemin_xyz.parent.parent / "MNT5m"
        if dossier_mnt.is_dir():
            correspondances = sorted(dossier_mnt.glob(f"*{dalle_id}_MNT5_*.asc"))
            if correspondances:
                return correspondances[0]

    return None

# 3. SÉLECTION DES CANDIDATS À INSPECTER
def calculer_candidats(
    donnees: dict[str, np.ndarray],
    methodes: list[str],
    fraction_alertes: float,
    ecart_vertical_min: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Choisit les points avec les scores les plus élevés, méthode par méthode.

    Étapes :
      1. Ne considérer que les sondages éligibles.
      2. Fixer un seuil de score selon le budget d'alertes de l'interface.
      3. Exiger un résidu vertical absolu minimal par rapport au plan local.
      4. Compter combien de méthodes signalent chaque point (« votes »).

    ATTENTION : ce seuil est choisi pour la *visualisation*, sur chaque dalle.
    Il ne remplace pas les seuils figés du benchmark de test.
    """
    eligibles = donnees["eligible"].astype(bool)
    residus = donnees["plane_residual_m"]

    indicateurs = {}
    votes = np.zeros(len(eligibles), dtype=np.int16)

    for methode in methodes:
        if not np.any(eligibles):
            indicateurs[methode] = np.zeros(len(eligibles), dtype=bool)
            continue

        scores = donnees[methode]
        seuil_score = float(np.quantile(scores[eligibles], 1 - fraction_alertes))

        candidats = (
            eligibles
            & (scores >= seuil_score)
            & (np.abs(residus) >= ecart_vertical_min)
        )

        indicateurs[methode] = candidats
        votes += candidats.astype(np.int16)

    return indicateurs, votes


def indices_aleatoires(indices: np.ndarray, maximum: int, graine: int = 42):
    """Échantillonne des points pour limiter le coût d'affichage de Plotly."""
    if len(indices) == 0:
        return np.array([], dtype=int)

    generateur = np.random.default_rng(graine)
    return generateur.choice(indices, size=min(len(indices), maximum), replace=False)

# 4. CARTE 2D : LOCALISATION XY SUR PLUSIEURS DALLES
def construire_carte_xy(
    dalles: list[str],
    nuages: dict[str, dict],
    methodes: list[str],
    fraction_alertes: float,
    ecart_vertical_min: float,
    points_par_dalle: int,
) -> go.Figure:
    """Superpose les dalles en XY (Lambert-93) et leurs candidats colorés."""
    figure = go.Figure()

    # Fond de carte : un sous-échantillon discret des sondages de chaque dalle.
    for dalle_id in dalles:
        donnees = nuages[dalle_id]
        xyz = donnees["xyz"]
        eligibles = np.flatnonzero(donnees["eligible"])
        selection = indices_aleatoires(eligibles, points_par_dalle)

        figure.add_trace(
            go.Scattergl(
                x=xyz[selection, 0],
                y=xyz[selection, 1],
                mode="markers",
                name=f"{dalle_id} — points",
                legendgroup="terrain",
                showlegend=(dalle_id == dalles[0]),
                text=[dalle_id] * len(selection),
                hovertemplate=(
                    "Dalle %{text}<br>X %{x:.1f} m<br>Y %{y:.1f} m"
                    "<extra></extra>"
                ),
                marker=dict(size=3, opacity=0.30, color="#8297ab"),
            )
        )

    # Un calque distinct par algorithme facilite les comparaisons dans la légende.
    for methode in methodes:
        coordonnees_x, coordonnees_y, identifiants = [], [], []

        for dalle_id in dalles:
            donnees = nuages[dalle_id]
            candidats, _ = calculer_candidats(
                donnees, [methode], fraction_alertes, ecart_vertical_min
            )
            points = donnees["xyz"][candidats[methode]]

            coordonnees_x.extend(points[:, 0])
            coordonnees_y.extend(points[:, 1])
            identifiants.extend([dalle_id] * len(points))

        figure.add_trace(
            go.Scattergl(
                x=coordonnees_x,
                y=coordonnees_y,
                text=identifiants,
                mode="markers",
                name=NOMS_METHODES[methode],
                hovertemplate=(
                    "Dalle %{text}<br>X %{x:.1f} m<br>Y %{y:.1f} m"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=8,
                    color=COULEURS_METHODES[methode],
                    opacity=0.85,
                    symbol="circle-open",
                    line=dict(width=2),
                ),
            )
        )

    figure.update_layout(
        title="Localisation XY des candidats (Lambert-93, EPSG:2154)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=580,
        margin=dict(l=5, r=5, t=45, b=10),
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure

# 5. RELIEF 3D : UNE FIGURE INDÉPENDANTE PAR DALLE
def altitudes_du_mnt_aux_points(
    points: np.ndarray,
    grille_z: np.ndarray,
    coefficients_affines: tuple,
    pas: int,
) -> np.ndarray:
    """Retrouve la hauteur du pixel MNT affiché le plus proche de chaque XYZ.

    Cette fonction sert UNIQUEMENT à positionner visuellement les marqueurs
    au-dessus du relief, pour qu'ils ne soient pas cachés sous la surface.
    Les vraies altitudes des sondages restent intactes dans les exports.
    """
    if len(points) == 0:
        return np.empty(0, dtype=float)

    transformation = Affine(*coefficients_affines)
    colonnes, lignes = (~transformation) * (points[:, 0], points[:, 1])

    # Le raster d'affichage a éventuellement été réduit par un facteur « pas ».
    colonnes = np.floor(np.asarray(colonnes) / pas).astype(int)
    lignes = np.floor(np.asarray(lignes) / pas).astype(int)

    dans_la_grille = (
        (colonnes >= 0)
        & (colonnes < grille_z.shape[1])
        & (lignes >= 0)
        & (lignes < grille_z.shape[0])
    )

    altitudes = np.full(len(points), np.nan, dtype=float)
    altitudes[dans_la_grille] = grille_z[
        lignes[dans_la_grille], colonnes[dans_la_grille]
    ]
    return altitudes


def ajouter_surface_mnt(
    figure: go.Figure,
    dalle_id: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    origine: tuple[float, float, float],
    exageration: int,
    palette: str,
    courbes_niveau: bool,
) -> None:
    """Ajoute la grille bathymétrique réelle à une scène Plotly 3D."""
    origine_x, origine_y, origine_z = origine

    # Les coordonnées absolues restent disponibles dans les infobulles.
    grille_x = np.broadcast_to(x[None, :], z.shape)
    grille_y = np.broadcast_to(y[:, None], z.shape)
    informations_survol = np.stack((grille_x, grille_y, z), axis=-1)

    figure.add_trace(
        go.Surface(
            x=x - origine_x,
            y=y - origine_y,
            z=(z - origine_z) * exageration,
            surfacecolor=z,  # La couleur représente le Z original, non amplifié.
            colorscale=palette,
            cmin=float(np.nanmin(z)),
            cmax=float(np.nanmax(z)),
            customdata=informations_survol,
            opacity=0.99,
            name=f"MNT5m {dalle_id}",
            showscale=True,
            colorbar=dict(title="Altitude Z (m)", len=0.64, thickness=14),
            contours=dict(
                z=dict(
                    show=courbes_niveau,
                    usecolormap=True,
                    highlightcolor="#f5f5f5",
                    width=1,
                )
            ),
            hovertemplate=(
                "MNT5m<br>X %{customdata[0]:.1f} m"
                "<br>Y %{customdata[1]:.1f} m"
                "<br>Altitude %{customdata[2]:.2f} m<extra></extra>"
            ),
            lighting=dict(ambient=0.65, diffuse=0.75, roughness=0.9, specular=0.1),
            lightposition=dict(x=500, y=200, z=900),
        )
    )


def ajouter_sondages_3d(
    figure: go.Figure,
    points: np.ndarray,
    origine: tuple[float, float, float],
    exageration: int,
    palette: str,
    afficher_echelle_couleurs: bool,
) -> None:
    """Ajoute les sondages (points d'origine, sans altération) à la scène."""
    origine_x, origine_y, origine_z = origine

    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0] - origine_x,
            y=points[:, 1] - origine_y,
            z=(points[:, 2] - origine_z) * exageration,
            mode="markers",
            name="Sondages XYZ (échantillon)",
            customdata=points,
            hovertemplate=(
                "Sondage original<br>X %{customdata[0]:.1f} m"
                "<br>Y %{customdata[1]:.1f} m"
                "<br>Z %{customdata[2]:.2f} m<extra></extra>"
            ),
            marker=dict(
                size=1.8,
                color=points[:, 2],
                colorscale=palette,
                opacity=0.5,
                showscale=afficher_echelle_couleurs,
            ),
        )
    )


def ajouter_candidats_3d(
    figure: go.Figure,
    donnees: dict[str, np.ndarray],
    methode: str,
    indices: np.ndarray,
    nombre_total: int,
    origine: tuple[float, float, float],
    exageration: int,
    terrain: tuple | None,
    indice_methode: int,
    pas_raster: int,
) -> None:
    """Affiche les candidats de la méthode, avec leurs vrais Z au survol."""
    points = donnees["xyz"][indices]
    if len(points) == 0:
        return

    origine_x, origine_y, origine_z = origine

    if terrain is None:
        altitudes_visuelles = points[:, 2]
    else:
        _, _, grille_z, coefficients = terrain
        hauteur_surface = altitudes_du_mnt_aux_points(
            points, grille_z, coefficients, pas_raster
        )

        # Décalage purement graphique pour ne pas masquer les points dans le MNT.
        # Chaque méthode reçoit un petit décalage supplémentaire.
        altitudes_visuelles = (
            np.where(np.isfinite(hauteur_surface), hauteur_surface, points[:, 2])
            + 0.6
            + indice_methode * 0.18
        )

    informations_survol = np.column_stack(
        (
            points,
            donnees[methode][indices],
            donnees["plane_residual_m"][indices],
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0] - origine_x,
            y=points[:, 1] - origine_y,
            z=(altitudes_visuelles - origine_z) * exageration,
            mode="markers",
            name=f"{NOMS_METHODES[methode]} ({nombre_total})",
            customdata=informations_survol,
            hovertemplate=(
                f"{NOMS_METHODES[methode]}"
                "<br>X %{customdata[0]:.1f} m"
                "<br>Y %{customdata[1]:.1f} m"
                "<br>Z sondage %{customdata[2]:.2f} m"
                "<br>Score %{customdata[3]:.3f}"
                "<br>Résidu plan %{customdata[4]:.2f} m<extra></extra>"
            ),
            marker=dict(
                size=5,
                color=COULEURS_METHODES[methode],
                opacity=0.97,
                line=dict(color="#20252c", width=0.8),
            ),
        )
    )


def construire_relief_3d(
    dalle_id: str,
    donnees: dict[str, np.ndarray],
    options: OptionsAffichage,
    mnt: Path | None,
) -> tuple[go.Figure, dict[str, int], bool, str | None]:
    """Construit UNE scène 3D pour UNE seule dalle.

    Si le MNT5m est disponible : surface réelle + anomalies candidates.
    Sinon : nuage XYZ uniquement, sans surface artificielle.
    """
    xyz = donnees["xyz"]
    eligibles = np.flatnonzero(donnees["eligible"])
    figure = go.Figure()

    terrain = None
    systeme_coordonnees = None

    if mnt is not None:
        x, y, z, coefficients, systeme_coordonnees = charger_mnt(
            str(mnt), options.pas_raster
        )
        if np.isfinite(z).any():
            terrain = (x, y, z, coefficients)

    if terrain is not None:
        x, y, z, _ = terrain
        origine = (float(np.min(x)), float(np.min(y)), float(np.nanmin(z)))
        ajouter_surface_mnt(
            figure,
            dalle_id,
            x,
            y,
            z,
            origine,
            options.exageration_verticale,
            options.palette,
            options.afficher_courbes,
        )
    else:
        # Important : les XYZ clairsemés ne doivent pas être transformés
        # implicitement en surface « mesurée ».
        origine = tuple(np.min(xyz, axis=0).astype(float))

    if terrain is None or options.afficher_sondages:
        selection = indices_aleatoires(eligibles, options.points_3d)
        ajouter_sondages_3d(
            figure,
            xyz[selection],
            origine,
            options.exageration_verticale,
            options.palette,
            afficher_echelle_couleurs=(terrain is None),
        )

    candidats, _ = calculer_candidats(
        donnees,
        options.methodes,
        options.fraction_alertes,
        options.ecart_vertical_min,
    )

    nombres_candidats = {}
    for indice_methode, methode in enumerate(options.methodes):
        indices = np.flatnonzero(candidats[methode])
        nombres_candidats[methode] = len(indices)

        # L'affichage est limité à 1 500 marqueurs par méthode ; les compteurs
        # et l'export CSV continuent à inclure tous les candidats.
        indices_affiches = indices[:1500]
        ajouter_candidats_3d(
            figure,
            donnees,
            methode,
            indices_affiches,
            nombres_candidats[methode],
            origine,
            options.exageration_verticale,
            terrain,
            indice_methode,
            options.pas_raster,
        )

    figure.update_layout(
        title=f"Dalle {dalle_id} — relief 3D du MNT5m et candidats XYZ",
        height=620,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", y=-0.02),
        scene=dict(
            xaxis_title=f"X depuis {origine[0]:.0f} m",
            yaxis_title=f"Y depuis {origine[1]:.0f} m",
            zaxis_title=(
                f"(Z − {origine[2]:.1f} m) ×{options.exageration_verticale}"
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.65, z=0.95)),
        ),
    )

    return figure, nombres_candidats, terrain is not None, systeme_coordonnees

# 6. EXPORT CSV DES POINTS SUSPECTS
def preparer_export_candidats(
    nuages: dict[str, dict], options: OptionsAffichage
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Construit une ligne CSV par point signalé par au moins une méthode.

    Le nombre de « votes » correspond au nombre de méthodes qui signalent
    ce même sondage XYZ. Cela ne constitue pas une validation terrain.
    """
    lignes = []
    totaux = {methode: 0 for methode in options.methodes}
    nombre_consensus = 0

    for dalle_id, donnees in nuages.items():
        indicateurs, votes = calculer_candidats(
            donnees,
            options.methodes,
            options.fraction_alertes,
            options.ecart_vertical_min,
        )

        for methode in options.methodes:
            totaux[methode] += int(indicateurs[methode].sum())

        nombre_consensus += int((votes >= 2).sum())

        for indice in np.flatnonzero(votes >= 1):
            x, y, z = donnees["xyz"][indice]
            ligne = {
                "tile_id": dalle_id,
                "point_index": int(indice),
                "x_lambert93_m": float(x),
                "y_lambert93_m": float(y),
                "z_ign69_m": float(z),
                "local_plane_residual_m": float(
                    donnees["plane_residual_m"][indice]
                ),
                "votes": int(votes[indice]),
            }

            for methode in options.methodes:
                ligne[f"{methode}_score"] = float(donnees[methode][indice])
                ligne[f"{methode}_flag"] = bool(indicateurs[methode][indice])

            lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    if not tableau.empty:
        tableau = tableau.sort_values(
            ["votes", "tile_id"], ascending=[False, True]
        )

    return tableau, totaux, nombre_consensus

# 7. INTERFACE STREAMLIT : CONTRÔLES ET ONGLETS
def afficher_indicateurs(resume: dict, nombre_dalles: int) -> None:
    """Affiche les chiffres clés du benchmark pré-calculé."""
    st.info(
        "Les métriques synthétiques proviennent de sondages artificiellement "
        "déplacés en Z sur des dalles de test. Les autres sondages ne sont "
        "pas certifiés corrects : leur taux d'alertes n'est pas un vrai taux "
        "de faux positifs."
    )

    colonnes = st.columns(4)
    colonnes[0].metric(
        "Dalles de test évaluées", str(resume["test_tiles_evaluated"])
    )
    colonnes[1].metric(
        "Sondages injectés au test",
        f"{resume['test_injected_points']:,}".replace(",", " "),
    )
    colonnes[2].metric("Dalles affichables", nombre_dalles)
    colonnes[3].metric(
        "Calibration", f"{resume['nominal_alert_fraction'] * 100:.1f} % nominal"
    )


def afficher_benchmark(
    classement: pd.DataFrame,
    resultats_dalles: pd.DataFrame,
) -> None:
    """Présente les métriques globales puis les résultats détaillés par dalle."""
    st.subheader("Performance sur les dalles de test du benchmark")

    affichage = classement.copy()
    affichage["Modèle"] = affichage["method"].map(NOMS_METHODES)
    colonnes = [
        "Modèle",
        "roc_auc_synthetic",
        "average_precision_synthetic",
        "recall_injected",
        "alert_fraction_unchanged",
        "tile_recall_sd",
    ]
    traductions = {
        "roc_auc_synthetic": "ROC-AUC synthétique",
        "average_precision_synthetic": "AP synthétique",
        "recall_injected": "Rappel sur injectées",
        "alert_fraction_unchanged": "Alertes sur inchangés",
        "tile_recall_sd": "Écart-type rappel (dalles)",
    }

    st.dataframe(
        affichage[colonnes].rename(columns=traductions),
        hide_index=True,
        use_container_width=True,
    )

    graphique = px.bar(
        affichage,
        x="Modèle",
        y="recall_injected",
        text="recall_injected",
        labels={"recall_injected": "Rappel sur anomalies injectées"},
        range_y=[0, 1],
    )
    graphique.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    st.plotly_chart(graphique, use_container_width=True)

    st.subheader("Résultats détaillés par dalle")
    methode = st.selectbox(
        "Méthode (résultats par dalle)",
        METHODES,
        index=3,
        format_func=lambda nom: NOMS_METHODES[nom],
    )
    colonnes_dalles = [
        "tile_id",
        f"{methode}_roc_auc_synthetic",
        f"{methode}_average_precision_synthetic",
        f"{methode}_recall_injected",
        f"{methode}_alert_fraction_unchanged",
    ]
    st.dataframe(
        resultats_dalles[colonnes_dalles],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Télécharger le benchmark CSV",
        data=resultats_dalles.to_csv(index=False).encode("utf-8"),
        file_name="hydroqc_benchmark_multidalles.csv",
        mime="text/csv",
    )


def afficher_carte_xy(nuages: dict[str, dict], options: OptionsAffichage) -> None:
    """Affiche une carte XY commune aux dalles sélectionnées."""
    figure = construire_carte_xy(
        options.dalles,
        nuages,
        options.methodes,
        options.fraction_alertes,
        options.ecart_vertical_min,
        options.points_2d,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Coordonnées Lambert-93 (m). Les positions proviennent des sondages XYZ."
    )


def afficher_cartes_3d(
    nuages: dict[str, dict],
    options: OptionsAffichage,
    chemins_mnt: dict[str, str],
    resume: dict,
) -> None:
    """Affiche successivement une carte Plotly indépendante pour chaque dalle."""
    st.info(
        "Chaque dalle possède sa propre scène 3D. Le terrain est le MNT5m "
        "Litto3D réel, et non une interpolation artificielle des XYZ."
    )

    for dalle_id in options.dalles:
        mnt = trouver_mnt(dalle_id, chemins_mnt, resume)
        figure, comptes, relief_present, crs = construire_relief_3d(
            dalle_id, nuages[dalle_id], options, mnt
        )
        st.plotly_chart(figure, use_container_width=True, key=f"landscape_{dalle_id}")

        if relief_present:
            st.caption(
                f"{dalle_id} · Surface MNT5m : {mnt} · "
                f"Référentiel horizontal : {crs or 'Lambert-93 (nom du fichier)'}. "
                "Les marqueurs sont légèrement rehaussés pour rester visibles ; "
                "les infobulles et les CSV conservent l'altitude XYZ originale."
            )
        else:
            st.warning(
                f"{dalle_id} : MNT5m introuvable. Affichage en nuage XYZ. "
                "Vérifie le manifeste ou le dossier data/mnt5m."
            )

        st.caption(
            "Candidats : "
            + ", ".join(
                f"{NOMS_METHODES[methode]} : {comptes[methode]}"
                for methode in options.methodes
            )
        )
        st.divider()

    st.caption(
        "L'exagération verticale ne modifie ni les détections, ni les altitudes "
        "originales exportées."
    )


def afficher_export_candidats(
    nuages: dict[str, dict], options: OptionsAffichage
) -> None:
    """Montre les candidats et propose leur téléchargement au format CSV."""
    tableau, totaux, nombre_consensus = preparer_export_candidats(nuages, options)

    colonnes = st.columns(3)
    colonnes[0].metric("Candidats distincts", len(tableau))
    colonnes[1].metric("Accord ≥2 méthodes", nombre_consensus)
    colonnes[2].metric("Dalles inspectées", len(options.dalles))

    st.write(
        "Détections par méthode : "
        + ", ".join(
            f"{NOMS_METHODES[methode]} : {nombre}"
            for methode, nombre in totaux.items()
        )
    )

    if tableau.empty:
        st.warning("Aucun candidat pour les seuils actuels.")
        return

    st.dataframe(
        tableau, hide_index=True, use_container_width=True, height=450
    )
    st.download_button(
        "Exporter les candidats sélectionnés en CSV",
        data=tableau.to_csv(index=False).encode("utf-8"),
        file_name="hydroqc_3d_candidats.csv",
        mime="text/csv",
    )

# 8. POINT D'ENTRÉE DE L'APPLICATION
def afficher_barre_laterale(dalles_disponibles: list[str]) -> OptionsAffichage:
    """Configure les filtres (le dossier de résultats est déjà demandé dans main)."""
    st.sidebar.subheader("Cartes et comparaison")
    dalles = st.sidebar.multiselect(
        "Dalles affichées",
        dalles_disponibles,
        default=dalles_disponibles[: min(3, len(dalles_disponibles))],
        max_selections=12,
    )
    methodes = st.sidebar.multiselect(
        "Algorithmes",
        list(METHODES),
        default=["plane", "iforest", "pointnet"],
        format_func=lambda methode: NOMS_METHODES[methode],
    )
    budget_pourcent = st.sidebar.slider(
        "Budget de revue par méthode (%)", 0.1, 3.0, 0.5, 0.1
    )
    ecart_vertical_min = st.sidebar.slider(
        "Écart vertical minimal (m)", 0.0, 3.0, 0.5, 0.1
    )
    points_2d = st.sidebar.slider(
        "Points terrain / dalle (2D)", 500, 10000, 4000, 500
    )
    points_3d = st.sidebar.slider(
        "Points terrain / dalle (3D)", 500, 6000, 2000, 500
    )
    exageration = st.sidebar.slider("Exagération verticale (3D)", 1, 80, 25)

    st.sidebar.subheader("Relief 3D par dalle")
    manifeste = chemin_application(
        st.sidebar.text_input(
            "Manifeste XYZ (pour retrouver les MNT)",
            "outputs/xyz3d/manifest.json",
        )
    )
    pas_raster = st.sidebar.select_slider(
        "Résolution d'affichage du MNT",
        options=[1, 2, 4],
        value=1,
        format_func=lambda pas: f"1 pixel sur {pas}",
    )
    sondages = st.sidebar.checkbox(
        "Superposer un échantillon des sondages XYZ", False
    )
    courbes = st.sidebar.checkbox("Afficher les courbes de niveau", False)
    palette = st.sidebar.selectbox(
        "Palette du relief", ["Viridis", "Earth", "Turbo", "Cividis"], index=0
    )

    return OptionsAffichage(
        dalles=dalles,
        methodes=methodes,
        fraction_alertes=budget_pourcent / 100,
        ecart_vertical_min=ecart_vertical_min,
        points_2d=points_2d,
        points_3d=points_3d,
        exageration_verticale=exageration,
        manifeste=manifeste,
        pas_raster=pas_raster,
        afficher_sondages=sondages,
        afficher_courbes=courbes,
        palette=palette,
    )


def main() -> None:
    """Orchestre les chargements, les contrôles et les quatre onglets."""
    st.set_page_config(
        page_title="HydroQC AI | Bilan final", page_icon="🌊", layout="wide"
    )
    st.title("HydroQC AI : comparaison Machine Learning / Deep Learning")
    st.caption(
        "Sondages Litto3D XYZ · contrôle qualité exploratoire · "
        "détections ≠ erreurs confirmées"
    )

    # On récupère d'abord la racine pour déterminer les dalles disponibles.
    # Un text_input avec la même clé implicite n'est pas utilisé deux fois :
    # la sélection du dossier reste donc déclarée ici, avant les autres filtres.
    racine = chemin_application(
        st.sidebar.text_input("Répertoire des résultats", "outputs/hydroqc_final")
    )

    if not racine.is_dir():
        st.error(
            f"Répertoire absent : {racine}. "
            "Vérifie les fichiers exportés dans outputs/hydroqc_final."
        )
        st.stop()

    manquants = [nom for nom in FICHIERS_BENCHMARK if not (racine / nom).is_file()]
    if manquants:
        st.error(f"Fichiers de benchmark manquants : {', '.join(manquants)}")
        st.stop()

    fichiers_scores = {
        fichier.stem.removeprefix("scores_"): fichier
        for fichier in sorted(racine.glob("scores_*.npz"))
    }
    if not fichiers_scores:
        st.error("Aucun fichier scores_<dalle>.npz dans le dossier de résultats.")
        st.stop()

    classement = charger_csv(str(racine / "leaderboard.csv"))
    resultats_dalles = charger_csv(str(racine / "synthetic_per_tile.csv"))
    resume = json.loads((racine / "run_summary.json").read_text(encoding="utf-8"))

    options = afficher_barre_laterale(list(fichiers_scores))

    try:
        chemins_mnt = charger_chemins_mnt(options.manifeste)
    except (OSError, ValueError, TypeError) as erreur:
        chemins_mnt = {}
        st.warning(f"Lecture du manifeste impossible : {erreur}")

    afficher_indicateurs(resume, len(fichiers_scores))

    onglet_benchmark, onglet_xy, onglet_3d, onglet_export = st.tabs(
        [
            "Benchmark synthétique",
            "Carte XY multi-dalles",
            "Relief 3D par dalle",
            "Candidats & export",
        ]
    )

    with onglet_benchmark:
        afficher_benchmark(classement, resultats_dalles)

    if not options.dalles or not options.methodes:
        for onglet in (onglet_xy, onglet_3d, onglet_export):
            with onglet:
                st.warning("Sélectionne au moins une dalle et une méthode.")
    else:
        # Les scores de chaque dalle sont chargés une fois par réexécution.
        nuages = {
            dalle_id: charger_scores_xyz(str(fichiers_scores[dalle_id]))
            for dalle_id in options.dalles
        }

        with onglet_xy:
            afficher_carte_xy(nuages, options)
        with onglet_3d:
            afficher_cartes_3d(nuages, options, chemins_mnt, resume)
        with onglet_export:
            afficher_export_candidats(nuages, options)

    st.divider()
    st.caption(
        "Méthodologie : blocs géographiques train/val/test ; validation "
        "scindée en optimisation et calibration ; seuils du benchmark fixés "
        "avant le test ; sondages réels non étiquetés. "
        "Ce prototype n'est pas un logiciel de validation hydrographique certifié."
    )




if __name__ == "__main__":
    main()
