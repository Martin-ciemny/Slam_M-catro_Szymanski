"""TP1 - Construction d'une Grid Map 2D et Localisation Brute Force (Simple)."""
import argparse
import math
import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scan_io import Scan, load_photo_csv, load_scan
from viewer import apply_saved_poses

# =====================================================================
# 1. OUTILS GÉOMÉTRIQUES ET STRUCTURE DE LA CARTE
# =====================================================================
def bresenham(i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
    """Algorithme de tracé de ligne de Bresenham. Renvoie les cases traversées sans l'impact final."""
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
    def __init__(self, xmin=-12.0, xmax=12.0, ymin=-12.0, ymax=12.0, resolution=0.05):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.resolution = resolution
        
        # Calcul du nombre de cases de la grille
        self.nx = int(math.ceil((xmax - xmin) / resolution))
        self.ny = int(math.ceil((ymax - ymin) / resolution))
        
        # 127 = inconnu, 0 = libre, 255 = mur
        self.grid = np.full((self.ny, self.nx), 127, dtype=np.uint8)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convertit des mètres (monde) en indices de tableau (grille)."""
        i = int(math.floor((x - self.xmin) / self.resolution))
        j = int(math.floor((y - self.ymin) / self.resolution))
        return i, j

    def in_bounds(self, i: int, j: int) -> bool:
        """Vérifie que l'on ne sort pas du tableau."""
        return 0 <= i < self.nx and 0 <= j < self.ny

def mettre_a_jour_carte(carte: GridMap, scan: Scan):
    """Met à jour la carte avec un nouveau scan."""
    xr, yr, _ = scan.pose
    ir, jr = carte.world_to_grid(xr, yr)

    for x_imp, y_imp in scan.points_world():
        i_imp, j_imp = carte.world_to_grid(x_imp, y_imp)

        # 1. Le rayon traverse du vide : on diminue la valeur vers 0 (libre)
        for ic, jc in bresenham(ir, jr, i_imp, j_imp):
            if carte.in_bounds(ic, jc):
                val = int(carte.grid[jc, ic]) - 20
                carte.grid[jc, ic] = max(0, val)

        # 2. Le rayon tape un obstacle : on augmente la valeur vers 255 (mur)
        if carte.in_bounds(i_imp, j_imp):
            val = int(carte.grid[j_imp, i_imp]) + 45
            carte.grid[j_imp, i_imp] = min(255, val)

# =====================================================================
# 2. ALGORITHME DE LOCALISATION (SCAN MATCHING)
# =====================================================================
def evaluer_score(carte: GridMap, scan: Scan, x_test: float, y_test: float, theta_test: float) -> int:
    """Simule le robot à (x, y, theta) et compte les points Lidar qui touchent un mur."""
    score = 0
    angle_rad = math.radians(theta_test)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    for px, py in scan.points_local():
        x_monde = x_test + (px * cos_a - py * sin_a)
        y_monde = y_test + (px * sin_a + py * cos_a)
        
        i, j = carte.world_to_grid(x_monde, y_monde)
        if carte.in_bounds(i, j):
            if carte.grid[j, i] > 127:  # Si la case tend vers occupé
                score += int(carte.grid[j, i])
    return score

def localisation_brute_force(carte: GridMap, scan: Scan, pose_depart: tuple[float,float,float]):
    """Teste toutes les positions autour du départ pour trouver le score maximum."""
    x_dep, y_dep, th_dep = pose_depart
    meilleur_score = -1
    meilleure_pose = pose_depart
    
    # Recherche à +/- 20 cm et +/- 15 degrés
    for dx in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        for dy in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
            for dth in [-15, -10, -5, 0, 5, 10, 15]:
                x = x_dep + dx
                y = y_dep + dy
                theta = th_dep + dth
                
                score = evaluer_score(carte, scan, x, y, theta)
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_pose = (x, y, theta)
                    
    return meilleure_pose

# =====================================================================
# 3. LANCEMENT ET AFFICHAGE
# =====================================================================
def charger_csv_auto(chemin: str | Path, r_max: float = 12.0) -> Scan:
    p = Path(chemin)
    entete = ""
    for ligne in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if ligne.strip() and not ligne.startswith("#"):
            entete = ligne.strip()
            break
    if "Distance(mm)" in entete or (len(entete.split(",")) >= 5 and entete.split(",")[0].strip() == "0"):
        return load_photo_csv(p, r_max=r_max)
    return load_scan(p, r_max=r_max)

def afficher_carte(carte: GridMap, scans: list[Scan]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    extent = [carte.xmin, carte.xmax, carte.ymin, carte.ymax]
    
    # 1. Carte probabiliste brute
    ax1.imshow(carte.grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=255)
    ax1.set_title("Grid Map Probabiliste")
    
    # 2. Carte seuillée pour la navigation
    carte_seuillee = np.full_like(carte.grid, 127)
    carte_seuillee[carte.grid <= 80] = 0
    carte_seuillee[carte.grid >= 180] = 255
    ax2.imshow(carte_seuillee, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=255)
    ax2.set_title("Carte Seuillée (Navigation)")
    
    # Affichage des poses du robot
    cmap = plt.get_cmap("tab10")
    for idx, s in enumerate(scans):
        x0, y0, th = s.pose
        c = cmap(idx % 10)
        t = np.deg2rad(th)
        for ax in (ax1, ax2):
            # On met la légende uniquement sur la carte de droite pour ne pas surcharger
            label = s.name if ax == ax2 else ""
            ax.plot(x0, y0, "o", color=c, ms=6, label=label)
            ax.plot([x0, x0 + 0.25 * np.cos(t)], [y0, y0 + 0.25 * np.sin(t)], "-", color=c, lw=2)

    ax2.legend(loc="best", fontsize=8)
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="fichiers CSV")
    ap.add_argument("--photos", action="store_true")
    ap.add_argument("--poses", default="poses.json")
    args = ap.parse_args()

    # 1. Chargement des scans
    if args.photos:
        import panda2 as pm
        chemins = pm.liste_photos()
        scans = [charger_csv_auto(c) for c in chemins]
        poses = pm.calculer_poses([s.name for s in scans], pm.lire_deplacements())
        for s in scans:
            s.set_pose(*poses[s.name])
    else:
        paths = []
        for pattern in args.files:
            paths += sorted(glob(pattern)) or [pattern]
        scans = [charger_csv_auto(p) for p in paths]
        apply_saved_poses(scans, args.poses)

    # 2. Création de la carte
    tous_points = np.vstack([s.points_world() for s in scans] + [s.pose[:2] for s in scans])
    xmin, ymin = tous_points.min(axis=0) - 1.0
    xmax, ymax = tous_points.max(axis=0) + 1.0

    carte = GridMap(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, resolution=0.05)
    for s in scans:
        mettre_a_jour_carte(carte, s)

    # 3. Test de Localisation
    print("\n--- TEST LOCALISATION BRUTE FORCE ---")
    scan_test = scans[1] if len(scans) > 1 else scans[0]
    vraie_pose = scan_test.pose.copy()
    pose_bruitee = (vraie_pose[0] + 0.15, vraie_pose[1] + 0.10, vraie_pose[2] + 12.0)
    
    print(f"Départ erroné : x={pose_bruitee[0]:.2f}, y={pose_bruitee[1]:.2f}, th={pose_bruitee[2]:.1f}°")
    
    t0 = time.time()
    pose_cal = localisation_brute_force(carte, scan_test, pose_bruitee)
    t1 = time.time()
    
    print(f"Pose trouvée  : x={pose_cal[0]:.2f}, y={pose_cal[1]:.2f}, th={pose_cal[2]:.1f}°")
    print(f"Erreur de pos : {np.hypot(pose_cal[0]-vraie_pose[0], pose_cal[1]-vraie_pose[1])*100:.1f} cm")
    print(f"Temps calcul  : {t1 - t0:.3f} s\n")

    # Ajout des marqueurs pour l'affichage
    scan_faux = Scan(name="Pose Fausse", angle_deg=np.array([]), r_m=np.array([]))
    scan_faux.set_pose(*pose_bruitee)
    scans.append(scan_faux)
    
    scan_vrai = Scan(name="Pose Trouvée", angle_deg=np.array([]), r_m=np.array([]))
    scan_vrai.set_pose(*pose_cal)
    scans.append(scan_vrai)

    afficher_carte(carte, scans)

if __name__ == "__main__":
    main()