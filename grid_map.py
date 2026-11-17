"""TP1 - Étape 3 : Construction et mise à jour d'une Grid Map 2D (carte d'occupation).

Usage :
    python grid_map.py --photos                     construit la carte depuis photos/ + deplacements.csv
    python grid_map.py scan1.csv scan2.csv ...      construit la carte depuis des scans CSV
    python grid_map.py scans/*.csv --poses poses.json
    python grid_map.py --photos --res 0.05 --filter active le lissage spatial et la décroissance
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scan_io import Scan, load_photo_csv, load_scan
from viewer import apply_saved_poses


def bresenham(i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
    """Algorithme de tracé de ligne de Bresenham entre (i0, j0) et (i1, j1).
    Renvoie la liste des cellules traversées, en EXCLUANT la cellule d'impact (i1, j1).
    """
    cells = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj

    i, j = i0, j0
    while (i, j) != (i1, j1):
        cells.append((i, j))
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
    return cells


class GridMap:
    """Carte d'occupation 2D probabiliste stockée sur 8 bits (uint8 : 0=libre, 127=inconnu, 255=occupé)."""

    VAL_LIBRE = 0
    VAL_INCONNU = 127
    VAL_OCCUPE = 255

    def __init__(
        self,
        xmin: float = -6.0,
        xmax: float = 6.0,
        ymin: float = -6.0,
        ymax: float = 6.0,
        resolution: float = 0.05,
        inc_occupe: int = 45,
        dec_libre: int = 20,
    ):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.resolution = float(resolution)
        self.inc_occupe = int(inc_occupe)
        self.dec_libre = int(dec_libre)

        # Nombre de cellules selon X (colonnes) et Y (lignes)
        self.nx = int(np.ceil((self.xmax - self.xmin) / self.resolution))
        self.ny = int(np.ceil((self.ymax - self.ymin) / self.resolution))

        # État initial : tout inconnu (127)
        self.grid = np.full((self.ny, self.nx), self.VAL_INCONNU, dtype=np.uint8)
        # Propriétés supplémentaires : compteur d'impacts et timestamp (numéro du dernier scan)
        self.hits = np.zeros((self.ny, self.nx), dtype=np.uint16)
        self.last_seen = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.scan_count = 0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convertit des coordonnées monde (x, y) en mètres vers les indices (i, j) de la grille."""
        i = int(np.floor((x - self.xmin) / self.resolution))
        j = int(np.floor((y - self.ymin) / self.resolution))
        return i, j

    def in_bounds(self, i: int, j: int) -> bool:
        """Vérifie si la cellule (i, j) appartient aux limites de la grille."""
        return 0 <= i < self.nx and 0 <= j < self.ny

    def decroitre_temporellement(self, pas_oubli: int = 3):
        """Étape 5 : Décroissance temporelle des cellules occupées non réobservées."""
        mask_occ = (self.grid > self.VAL_INCONNU) & (self.last_seen < self.scan_count)
        self.grid[mask_occ] = np.maximum(
            self.VAL_INCONNU, self.grid[mask_occ].astype(np.int16) - pas_oubli
        ).astype(np.uint8)

    def filtrer_artefacts_isoles(self, seuil_occ: int = 180, min_voisins: int = 2):
        """Étape 5 : Lissage spatial supprimant les points occupés isolés (bruit lidar)."""
        occ = (self.grid >= seuil_occ).astype(np.uint8)
        # Somme des 8 voisins par décalage matriciel
        voisins = np.zeros_like(occ)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                voisins[1:-1, 1:-1] += occ[1 + dy : self.ny - 1 + dy, 1 + dx : self.nx - 1 + dx]

        isoles = (occ[1:-1, 1:-1] == 1) & (voisins[1:-1, 1:-1] < min_voisins)
        vue_interne = self.grid[1:-1, 1:-1]
        vue_interne[isoles] = self.VAL_INCONNU


# ------------------------------------------------------------------ Outils demandés au TP
def mettre_a_jour_carte(carte: GridMap, scan: Scan, decroissance: bool = False) -> GridMap:
    """Prend une carte d'occupation existante, un scan + pose, et met à jour :
    - les cellules traversées par le faisceau (diminution de la probabilité d'occupation)
    - les cellules d'impact (augmentation de la probabilité d'occupation)
    """
    carte.scan_count += 1

    # Position du robot dans le monde puis dans la grille
    xr, yr, _ = scan.pose
    ir, jr = carte.world_to_grid(xr, yr)
    if not carte.in_bounds(ir, jr):
        raise ValueError(f"La pose du robot ({xr:.2f}, {yr:.2f}) est hors de la grille.")

    # Coordonnées globales des points d'impact du scan
    pts_monde = scan.points_world()

    for x_imp, y_imp in pts_monde:
        i_imp, j_imp = carte.world_to_grid(x_imp, y_imp)

        # 1. Cellules traversées avant l'impact -> libres
        for ic, jc in bresenham(ir, jr, i_imp, j_imp):
            if carte.in_bounds(ic, jc):
                val = int(carte.grid[jc, ic]) - carte.dec_libre
                carte.grid[jc, ic] = max(GridMap.VAL_LIBRE, val)
                carte.last_seen[jc, ic] = carte.scan_count

        # 2. Cellule d'impact -> occupée
        if carte.in_bounds(i_imp, j_imp):
            val = int(carte.grid[j_imp, i_imp]) + carte.inc_occupe
            carte.grid[j_imp, i_imp] = min(GridMap.VAL_OCCUPE, val)
            carte.hits[j_imp, i_imp] += 1
            carte.last_seen[j_imp, i_imp] = carte.scan_count

    if decroissance:
        carte.decroitre_temporellement()

    return carte


def creer_carte_depuis_scan(
    scan: Scan,
    resolution: float = 0.05,
    marge: float = 1.0,
    limites: tuple[float, float, float, float] | None = None,
) -> GridMap:
    """Transforme un unique scan + pose en une nouvelle carte d'occupation."""
    if limites is None:
        pts = np.vstack([scan.points_world(), scan.pose[:2]])
        xmin, ymin = pts.min(axis=0) - marge
        xmax, ymax = pts.max(axis=0) + marge
    else:
        xmin, xmax, ymin, ymax = limites

    carte = GridMap(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, resolution=resolution)
    return mettre_a_jour_carte(carte, scan)


# ------------------------------------------------------------------ Affichage
def afficher_carte(carte: GridMap, scans: list[Scan], chemin_image: Path | None = None):
    """Affiche la carte d'occupation probabiliste et la trajectoire/poses des scans."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    extent = [carte.xmin, carte.xmax, carte.ymin, carte.ymax]

    # Vue 1 : Carte probabiliste (0 = blanc/libre, 127 = gris/inconnu, 255 = noir/occupé)
    im1 = ax1.imshow(
        carte.grid,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Occupation (0=libre, 127=inconnu, 255=occupé)")
    ax1.set_title(f"Grid Map 2D probabiliste (résolution = {carte.resolution*100:.0f} cm)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.25)

    # Vue 2 : Carte seuillée (Obstacles confirmés vs Espace libre vs Inconnu)
    carte_seuillee = np.full_like(carte.grid, 127)
    carte_seuillee[carte.grid <= 80] = 0
    carte_seuillee[carte.grid >= 180] = 255
    ax2.imshow(
        carte_seuillee,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax2.set_title("Carte seuillée (navigation) + Poses du robot")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.25)

    # Tracé des poses du robot sur les deux cartes
    cmap = plt.get_cmap("tab10")
    for idx, s in enumerate(scans):
        if not s.visible:
            continue
        x0, y0, th = s.pose
        c = cmap(idx % 10)
        t = np.deg2rad(th)
        for ax in (ax1, ax2):
            ax.plot(x0, y0, "o", color=c, markeredgecolor="k", ms=7, label=s.name if ax is ax2 else None)
            ax.plot([x0, x0 + 0.25 * np.cos(t)], [y0, y0 + 0.25 * np.sin(t)], "-", color=c, lw=2)

    if len(scans) <= 12:
        ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if chemin_image:
        plt.savefig(chemin_image, dpi=150)
        print(f"Carte d'occupation enregistrée dans : {chemin_image}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", help="fichiers CSV de scans")
    ap.add_argument("--photos", action="store_true", help="utilise les scans de photos/ et deplacements.csv")
    ap.add_argument("--lidar-cw", action="store_true", help="inverse les angles horaires du LD06")
    ap.add_argument("--poses", default="poses.json", help="fichier JSON des poses recalées avec viewer.py")
    ap.add_argument("--res", type=float, default=0.05, help="résolution d'une cellule en mètres (défaut: 0.05)")
    ap.add_argument("--rmax", type=float, default=12.0, help="portée maximale du lidar en mètres")
    ap.add_argument("--filter", action="store_true", help="active la décroissance temporelle et le filtrage spatial")
    args = ap.parse_args()

    # 1. Chargement des scans (compatible avec panda2.py et viewer.py)
    if args.photos:
        import panda2 as pm
        chemins = pm.liste_photos()
        if not chemins:
            ap.error(f"Aucune photo trouvée dans {pm.DOSSIER}")
        scans = [load_photo_csv(c, r_max=args.rmax) for c in chemins]
        poses = pm.calculer_poses([s.name for s in scans], pm.lire_deplacements())
        for s in scans:
            s.set_pose(*poses[s.name])
        out_img = pm.DOSSIER / "grid_map.png"
    else:
        if not args.files:
            ap.error("Indiquez des fichiers CSV ou utilisez l'option --photos")
        paths = []
        for pattern in args.files:
            paths += sorted(glob(pattern)) or [pattern]
        scans = [load_scan(p, lidar_cw=args.lidar_cw, r_max=args.rmax) for p in paths]
        apply_saved_poses(scans, args.poses)
        out_img = Path("grid_map.png")

    # 2. Calcul automatique de l'emprise globale de la grille
    tous_points = np.vstack([s.points_world() for s in scans] + [s.pose[:2] for s in scans])
    xmin, ymin = tous_points.min(axis=0) - 1.0
    xmax, ymax = tous_points.max(axis=0) + 1.0

    # 3. Initialisation avec le 1er scan puis mises à jour successives
    carte = creer_carte_depuis_scan(
        scans[0], resolution=args.res, limites=(xmin, xmax, ymin, ymax)
    )
    for s in scans[1:]:
        mettre_a_jour_carte(carte, s, decroissance=args.filter)

    if args.filter:
        carte.filtrer_artefacts_isoles()

    afficher_carte(carte, scans, out_img)


if __name__ == "__main__":
    main()