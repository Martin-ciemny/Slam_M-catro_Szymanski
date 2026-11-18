"""TP1 - Étapes 4 et 5 : Extraction de segments et Carte Vectorielle 2D.

Usage :
    python vector_map.py --photos
    python vector_map.py scan1.csv scan2.csv --poses poses.json
"""
import argparse
import math
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scan_io import Scan, load_photo_csv, load_scan
from viewer import apply_saved_poses


# =====================================================================
# PARTIE 4 : BOÎTE À OUTILS GÉOMÉTRIQUE (Manipulation de segments)
# =====================================================================
class Segment:
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

    @property
    def length(self) -> float:
        return np.linalg.norm(self.p2 - self.p1)

    @property
    def direction(self) -> np.ndarray:
        l = self.length
        return (self.p2 - self.p1) / l if l > 0 else np.array([0.0, 0.0])

    def angle_avec(self, autre: 'Segment') -> float:
        """Calcule l'angle (en radians) entre ce segment et un autre."""
        dot_product = np.clip(np.dot(self.direction, autre.direction), -1.0, 1.0)
        # On renvoie l'angle le plus petit (droites non orientées)
        angle = math.acos(dot_product)
        return min(angle, math.pi - angle)

    def est_proche(self, autre: 'Segment', tol_dist: float = 0.4, tol_angle_deg: float = 20.0) -> bool:
        """Détermine si deux segments sont proches angulairement et spatialement."""
        if math.degrees(self.angle_avec(autre)) > tol_angle_deg:
            return False
        
        # Calcul de la distance du centre de 'autre' à notre droite
        center_autre = (autre.p1 + autre.p2) / 2.0
        v = self.direction
        n = np.array([-v[1], v[0]]) # Vecteur normal
        dist_ortho = abs(np.dot(center_autre - self.p1, n))
        
        # Vérification de la distance des extrémités pour le recouvrement
        dist_p1 = min(np.linalg.norm(self.p1 - autre.p1), np.linalg.norm(self.p1 - autre.p2))
        dist_p2 = min(np.linalg.norm(self.p2 - autre.p1), np.linalg.norm(self.p2 - autre.p2))
        
        return dist_ortho < tol_dist and (dist_p1 < tol_dist * 3 or dist_p2 < tol_dist * 3)

    def fusionner(self, autre: 'Segment') -> 'Segment':
        """Fusionne deux segments proches en projetant les points sur la droite moyenne."""
        points = [self.p1, self.p2, autre.p1, autre.p2]
        # On prend les deux points les plus éloignés pour former le nouveau segment
        max_d = 0
        best_pair = (self.p1, self.p2)
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_d:
                    max_d = d
                    best_pair = (points[i], points[j])
        return Segment(best_pair[0], best_pair[1])

    def intersection(self, autre: 'Segment', tol: float = 1e-5) -> np.ndarray | None:
        """Calcule l'intersection stricte entre deux segments."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = autre.p1
        x4, y4 = autre.p2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < tol:  # Segments colinéaires ou parallèles
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        # L'intersection doit être STRICTEMENT à l'intérieur (on exclut 0 et 1 via tol)
        if tol < t < 1 - tol and tol < u < 1 - tol:
            return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
        return None

    def decouper(self, coupant: 'Segment', tol: float = 1e-3) -> 'Segment':
        """Découpe le segment s'il est traversé par un autre, et garde le plus long."""
        inter = self.intersection(coupant, tol)
        if inter is None:
            return self  # Pas de découpe (colinéaire ou intersection aux extrémités)

        seg1 = Segment(self.p1, inter)
        seg2 = Segment(inter, self.p2)

        l1, l2 = seg1.length, seg2.length
        if abs(l1 - l2) < tol:
            return seg1  # Si longueurs égales, on garde celui qui contient p1
        return seg1 if l1 > l2 else seg2


# =====================================================================
# PARTIE 5 : CONSTRUCTION DE LA CARTE VECTORIELLE
# =====================================================================
def douglas_peucker(points: np.ndarray, epsilon: float) -> list[Segment]:
    """Algorithme de Douglas-Peucker (Split-and-Merge) pour extraire des segments d'une courbe."""
    if len(points) < 2:
        return []
    
    p1 = points[0]
    p2 = points[-1]
    
    # Trouver le point le plus éloigné du segment [p1, p2]
    dmax = 0.0
    index = 0
    
    v = p2 - p1
    len_v = np.linalg.norm(v)
    if len_v == 0:
        return []
        
    n = np.array([-v[1], v[0]]) / len_v  # Vecteur normal unitaire
    
    for i in range(1, len(points) - 1):
        d = abs(np.dot(points[i] - p1, n))
        if d > dmax:
            index = i
            dmax = d
            
    # Si la distance maximale dépasse la tolérance, on coupe le cluster en deux au niveau de l'angle
    if dmax > epsilon:
        res1 = douglas_peucker(points[:index+1], epsilon)
        res2 = douglas_peucker(points[index:], epsilon)
        return res1 + res2
    else:
        return [Segment(p1, p2)]


class CarteVectorielle:
    def __init__(self):
        self.segments: list[Segment] = []

    def extraire_segments_nuage(self, pts_monde: np.ndarray, seuil_dist: float = 0.4, min_pts: int = 8) -> list[Segment]:
        """Clusterisation, extraction par Douglas-Peucker et filtrage des petits obstacles."""
        if len(pts_monde) < 2:
            return []

        segments_extraits = []
        cluster_actuel = [pts_monde[0]]

        for i in range(1, len(pts_monde)):
            p_prev = pts_monde[i - 1]
            p_curr = pts_monde[i]
            
            # Regroupement si les points sont assez proches
            if np.linalg.norm(p_curr - p_prev) < seuil_dist:
                cluster_actuel.append(p_curr)
            else:
                # Si discontinuité, on traite le cluster s'il a assez de points
                if len(cluster_actuel) >= min_pts:
                    segs = douglas_peucker(np.array(cluster_actuel), epsilon=0.1)
                    # FILTRE : On ignore les segments trop petits (bruit, pieds de chaises)
                    segments_extraits.extend([s for s in segs if s.length > 0.4])
                cluster_actuel = [p_curr]

        # Traitement du dernier cluster restant en fin de boucle
        if len(cluster_actuel) >= min_pts:
            segs = douglas_peucker(np.array(cluster_actuel), epsilon=0.1)
            segments_extraits.extend([s for s in segs if s.length > 0.4])

        return segments_extraits

    def mettre_a_jour(self, scan: Scan, position_robot: tuple[float, float]):
        """Ajoute les nouveaux segments à la carte et fusionne ou découpe si besoin."""
        nouveaux_segments = self.extraire_segments_nuage(scan.points_world())
        pos_robot_arr = np.array(position_robot)

        for nv_seg in nouveaux_segments:
            fusionne = False
            for i, seg_existant in enumerate(self.segments):
                # Si le nouveau segment est proche d'un existant, on les fusionne
                if nv_seg.est_proche(seg_existant):
                    self.segments[i] = seg_existant.fusionner(nv_seg)
                    fusionne = True
                    break
            
            if not fusionne:
                self.segments.append(nv_seg)

        # Vérification des absences (Ray-tracing)
        # On trace des segments entre le robot et les points d'impact
        #rayons = [Segment(pos_robot_arr, p) for p in scan.points_world()]
        #segments_nettoyes = []
        
        #for seg in self.segments:
            #seg_courant = seg
            # On vérifie si un rayon du lidar "traverse" un segment existant (porte ouverte)
            #for rayon in rayons[::5]: # [::5] pour ne pas tester tous les rayons (optimisation)
            #    seg_courant = seg_courant.decouper(rayon)
            
            # On garde le segment seulement s'il est encore assez grand
            #if seg_courant.length > 0.1: 
             #   segments_nettoyes.append(seg_courant)
                
        #self.segments = segments_nettoyes


# =====================================================================
# AFFICHAGE ET LANCEMENT
# =====================================================================
def afficher_carte_vectorielle(carte: CarteVectorielle, scans: list[Scan]):
    plt.figure(figsize=(10, 8))
    
    # Affichage du nuage de points brut en fond léger
    for s in scans:
        if s.visible:
            pts = s.points_world()
            plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.1, color='gray')
            plt.plot(s.pose[0], s.pose[1], 'ro', ms=4)

    # Affichage des segments vectoriels (les murs)
    for seg in carte.segments:
        plt.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], 'b-', linewidth=3)
        plt.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], 'co', ms=4) # Extrémités

    plt.title(f"Carte Vectorielle ({len(carte.segments)} segments détectés)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="fichiers CSV")
    ap.add_argument("--photos", action="store_true")
    ap.add_argument("--poses", default="poses.json")
    args = ap.parse_args()

    # Chargement (similaire à la Grid Map)
    if args.photos:
        import panda2 as pm
        chemins = pm.liste_photos()
        scans = [load_photo_csv(c) for c in chemins]
        poses = pm.calculer_poses([s.name for s in scans], pm.lire_deplacements())
        for s in scans:
            s.set_pose(*poses[s.name])
    else:
        paths = []
        for pattern in args.files:
            paths += sorted(glob(pattern)) or [pattern]
        scans = [load_scan(p) for p in paths]
        apply_saved_poses(scans, args.poses)

    # Création de la carte vectorielle
    carte_vec = CarteVectorielle()
    for scan in scans:
        carte_vec.mettre_a_jour(scan, position_robot=(scan.pose[0], scan.pose[1]))

    afficher_carte_vectorielle(carte_vec, scans)

if __name__ == "__main__":
    main()