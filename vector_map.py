"""TP1 - Étapes 4 et 5 : extraction de segments et carte vectorielle 2D.

Usage :
    python vector_map.py --photos
    python vector_map.py scan1.csv scan2.csv --poses poses.json
    python vector_map.py --photos --min-obs 2          (ne garde que ce qui est vu sur >= 2 photos : plus propre, mais
                                                       perd les éléments vus d'une seule photo)

Pipeline d'un scan :
    points monde -> clusters (seuil adaptatif) -> découpe Douglas-Peucker
    -> ajustement par moindres carrés de chaque morceau -> filtrage (longueur, qualité)
    -> insertion dans la carte avec fusion pondérée des segments proches.
"""
import argparse
import math
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scan_io import Scan, load_photo_csv, load_scan, premier_tour

TOL = 1e-3             # tolérance numérique des opérations géométriques : 1 mm
SIN_PARALLELE = 1e-6   # |sin(angle)| en dessous duquel deux segments sont dits parallèles


# =====================================================================
# PARTIE 4 : BOÎTE À OUTILS GÉOMÉTRIQUE (manipulation de segments)
# =====================================================================
def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


class Segment:
    def __init__(self, p1, p2, n_obs: int = 1):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.n_obs = n_obs  # nombre d'observations fusionnées (compteur de confiance)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def direction(self) -> np.ndarray:
        l = self.length
        return (self.p2 - self.p1) / l if l > 0 else np.array([0.0, 0.0])

    def distance_droite(self, p: np.ndarray) -> float:
        """Distance orthogonale d'un point à la DROITE portée par le segment."""
        d = self.direction
        return abs(_cross(d, np.asarray(p, float) - self.p1))

    def angle_avec(self, autre: "Segment") -> float:
        """Angle (rad, dans [0, pi/2]) entre les droites portées par les deux segments."""
        c = np.clip(abs(np.dot(self.direction, autre.direction)), 0.0, 1.0)
        return math.acos(c)

    def _ecart(self, autre: "Segment", tol_dist, tol_angle_deg, tol_gap):
        """None si les segments ne sont pas proches, sinon un coût (plus petit = plus proche)."""
        if self.length == 0 or autre.length == 0:
            return None
        ang = math.degrees(self.angle_avec(autre))
        if ang > tol_angle_deg:
            return None
        # on compare le plus court à la droite du plus long (le plus court est « absorbé »)
        ref, cand = (self, autre) if self.length >= autre.length else (autre, self)
        dmax = max(ref.distance_droite(cand.p1), ref.distance_droite(cand.p2))
        if dmax > tol_dist:
            return None
        # recouvrement le long de ref : intervalle [0, L] pour ref, [b0, b1] pour cand
        d = ref.direction
        b = sorted(((cand.p1 - ref.p1) @ d, (cand.p2 - ref.p1) @ d))
        gap = max(b[0] - ref.length, -b[1], 0.0)   # 0 si les intervalles se recouvrent
        if gap > tol_gap:
            return None
        return dmax + 0.005 * ang + 0.1 * gap

    def est_proche(self, autre: "Segment", tol_dist: float = 0.12,
                   tol_angle_deg: float = 8.0, tol_gap: float = 0.30) -> bool:
        """Proches angulairement (<= tol_angle_deg), en position (extrémités du plus court à
        <= tol_dist de la droite du plus long) et le long de l'axe (trou <= tol_gap)."""
        return self._ecart(autre, tol_dist, tol_angle_deg, tol_gap) is not None

    def fusionner(self, autre: "Segment") -> "Segment":
        """Fusion pondérée : droite moyenne (direction et position pondérées par longueur x observations),
        puis projection des 4 extrémités sur cette droite ; on garde les deux extrêmes."""
        w1, w2 = self.length * self.n_obs, autre.length * autre.n_obs
        d1, d2 = self.direction, autre.direction
        if np.dot(d1, d2) < 0:        # une droite n'a pas de sens : on aligne les orientations
            d2 = -d2
        d = w1 * d1 + w2 * d2
        d = d / np.linalg.norm(d)
        m = (w1 * (self.p1 + self.p2) / 2 + w2 * (autre.p1 + autre.p2) / 2) / (w1 + w2)
        a = [float((p - m) @ d) for p in (self.p1, self.p2, autre.p1, autre.p2)]
        return Segment(m + min(a) * d, m + max(a) * d, self.n_obs + autre.n_obs)

    def intersection(self, autre: "Segment", tol: float = TOL):
        """Point d'intersection s'il appartient aux DEUX segments (extrémités comprises, à tol près).
        Segments parallèles ou colinéaires -> None."""
        d1, d2 = self.p2 - self.p1, autre.p2 - autre.p1
        l1, l2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
        if l1 == 0 or l2 == 0:
            return None
        den = _cross(d1, d2)
        if abs(den) <= SIN_PARALLELE * l1 * l2:
            return None
        w = autre.p1 - self.p1
        t = _cross(w, d2) / den        # position le long de self  (0 = p1, 1 = p2)
        u = _cross(w, d1) / den        # position le long de autre
        if -tol / l1 <= t <= 1 + tol / l1 and -tol / l2 <= u <= 1 + tol / l2:
            return self.p1 + float(np.clip(t, 0.0, 1.0)) * d1
        return None

    def decouper(self, coupant: "Segment", tol: float = TOL) -> "Segment":
        """Si `coupant` traverse self (intersection sur les deux segments ET strictement à
        l'intérieur de self, à plus de tol de chaque extrémité), garde le plus long côté.
        Longueurs égales à tol près -> côté contenant p1. Sinon self est renvoyé tel quel."""
        i = self.intersection(coupant, tol)
        if i is None:
            return self
        if np.linalg.norm(i - self.p1) <= tol or np.linalg.norm(i - self.p2) <= tol:
            return self                # intersection à une extrémité : rien à couper
        s1, s2 = Segment(self.p1, i, self.n_obs), Segment(i, self.p2, self.n_obs)
        if abs(s1.length - s2.length) <= tol:
            return s1
        return s1 if s1.length > s2.length else s2


# =====================================================================
# PARTIE 5 : CONSTRUCTION DE LA CARTE VECTORIELLE
# =====================================================================
@dataclass
class Params:
    # segmentation en clusters : coupure si écart > gap_base + gap_k * r * dtheta
    gap_base: float = 0.10
    gap_k: float = 3.0
    # extraction
    eps: float = 0.05        # tolérance Douglas-Peucker (m) : ~2 x le bruit du LD06
    min_pts: int = 6         # points minimum pour un morceau
    min_len: float = 0.60    # longueur minimale d'un segment gardé (m) : écarte pieds de chaises et petits objets
    rms_max: float = 0.05    # écart quadratique maxi des points à la droite ajustée (m)
    # fusion
    tol_dist: float = 0.20   # photos recalées à la main : quelques cm à 15 cm d'écart entre deux vues d'un mur
    tol_angle: float = 8.0
    tol_gap: float = 0.50


def douglas_peucker_indices(points: np.ndarray, eps: float) -> list[tuple[int, int]]:
    """Découpe une polyligne en morceaux quasi rectilignes ; renvoie des plages d'indices (début, fin)."""
    def rec(lo: int, hi: int):
        v = points[hi] - points[lo]
        l = float(np.linalg.norm(v))
        if hi - lo < 2 or l == 0:
            return [(lo, hi)]
        n = np.array([-v[1], v[0]]) / l
        d = np.abs((points[lo + 1:hi] - points[lo]) @ n)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = lo + 1 + k
            return rec(lo, m) + rec(m, hi)
        return [(lo, hi)]
    return rec(0, len(points) - 1)


def ajuster_segment(pts: np.ndarray) -> tuple[Segment, float]:
    """Droite des moindres carrés (axe principal) ; extrémités = projections des points extrêmes.
    Renvoie (segment, écart quadratique moyen)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    a = (pts - c) @ d
    res = (pts - c) @ np.array([-d[1], d[0]])
    rms = float(np.sqrt(np.mean(res ** 2)))
    return Segment(c + a.min() * d, c + a.max() * d), rms


class CarteVectorielle:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.segments: list[Segment] = []

    # ---- extraction ------------------------------------------------
    def clusters(self, scan: Scan) -> list[np.ndarray]:
        """Regroupe les points consécutifs (ordre du scan) appartenant au même objet.
        Seuil adaptatif : l'écart entre deux points voisins grandit avec la distance."""
        n = len(scan.r_m)
        if n < 2:
            return []
        pts = scan.points_world()
        ang = np.deg2rad(scan.angle_deg)
        r = scan.r_m

        def pas(a, b):   # pas angulaire entre points voisins, plafonné (trous de mesure)
            return np.abs(np.angle(np.exp(1j * (a - b))))
        dth = pas(ang[1:], ang[:-1])
        dth = np.minimum(dth, 3 * np.median(dth))
        seuil = self.p.gap_base + self.p.gap_k * np.minimum(r[1:], r[:-1]) * dth
        coupe = np.linalg.norm(pts[1:] - pts[:-1], axis=1) > seuil
        groupes = np.split(np.arange(n), np.where(coupe)[0] + 1)

        # le scan est circulaire : le dernier groupe peut prolonger le premier (passage 360° -> 0°)
        if len(groupes) > 1:
            d0 = min(pas(ang[0], ang[-1]), 3 * np.median(dth))
            s0 = self.p.gap_base + self.p.gap_k * min(r[0], r[-1]) * d0
            if np.linalg.norm(pts[0] - pts[-1]) <= s0:
                groupes[0] = np.concatenate([groupes[-1], groupes[0]])
                groupes.pop()
        return [pts[g] for g in groupes if len(g) >= self.p.min_pts]

    def extraire_segments(self, scan: Scan) -> list[Segment]:
        """Clusterisation, Douglas-Peucker, ajustement moindres carrés, filtrage."""
        out = []
        for cl in self.clusters(scan):
            for lo, hi in douglas_peucker_indices(cl, self.p.eps):
                morceau = cl[lo:hi + 1]
                if len(morceau) < self.p.min_pts:
                    continue
                seg, rms = ajuster_segment(morceau)
                if seg.length >= self.p.min_len and rms <= self.p.rms_max:
                    out.append(seg)
        return out

    # ---- mise à jour -----------------------------------------------
    def inserer(self, seg: Segment):
        """Ajoute seg ; tant qu'il existe un segment proche, on fusionne avec LE PLUS PROCHE
        (et on recommence : le segment fusionné peut en rejoindre d'autres)."""
        q = self.p
        while True:
            meilleur, cout_min = None, None
            for k, ex in enumerate(self.segments):
                c = ex._ecart(seg, q.tol_dist, q.tol_angle, q.tol_gap)
                if c is not None and (cout_min is None or c < cout_min):
                    meilleur, cout_min = k, c
            if meilleur is None:
                break
            seg = self.segments.pop(meilleur).fusionner(seg)
        self.segments.append(seg)

    def mettre_a_jour(self, scan: Scan, position_robot=None):
        """Extrait les segments du scan et les insère dans la carte (position_robot : réservé
        à la détection d'absences par lancer de rayons, non utilisée ici)."""
        for s in self.extraire_segments(scan):
            self.inserer(s)

    def nettoyer(self, min_obs: int = 1):
        """Supprime les segments trop courts ou vus moins de min_obs fois."""
        self.segments = [s for s in self.segments
                         if s.length >= self.p.min_len and s.n_obs >= min_obs]

    def sauver(self, chemin) -> None:
        """Fichier texte x1,y1,x2,y2 (un segment par ligne)."""
        lignes = ["# Carte vectorielle (segments) : x1,y1,x2,y2"]
        lignes += [f"{s.p1[0]:.4f},{s.p1[1]:.4f},{s.p2[0]:.4f},{s.p2[1]:.4f}" for s in self.segments]
        Path(chemin).write_text("\n".join(lignes) + "\n", encoding="utf-8")


# =====================================================================
# AFFICHAGE ET LANCEMENT
# =====================================================================
def afficher_carte_vectorielle(carte: CarteVectorielle, scans: list[Scan], png=None, montrer=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    for s in scans:
        if s.visible:
            pts = s.points_world()
            ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.15, color="gray")
            ax.plot(s.pose[0], s.pose[1], "ro", ms=4)

    for seg in carte.segments:
        multi = seg.n_obs >= 2
        ax.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], "-",
                color="tab:blue" if multi else "tab:orange", linewidth=3 if multi else 2)
        ax.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], "co", ms=4)

    ax.plot([], [], "-", color="tab:blue", lw=3, label="vu sur >= 2 observations")
    ax.plot([], [], "-", color="tab:orange", lw=2, label="vu une seule fois")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Carte vectorielle ({len(carte.segments)} segments)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if png:
        fig.savefig(png, dpi=150)
    if montrer:
        plt.show()


def charger_scans(args):
    """Renvoie (scans, dossier_de_sortie)."""
    if args.photos:
        try:
            import photo_multi as pm
        except ImportError:
            import panda2 as pm
        chemins = pm.liste_photos()
        if not chemins:
            raise SystemExit(f"Aucune photo dans {pm.DOSSIER}")
        scans = [load_photo_csv(c, r_max=args.rmax) for c in chemins]
        poses = pm.calculer_poses([s.name for s in scans], pm.lire_deplacements())
        for s in scans:
            s.set_pose(*poses[s.name])
        if not args.tous_les_tours:
            scans = [premier_tour(s) for s in scans]
        return scans, pm.DOSSIER
    if not args.files:
        raise SystemExit("Indiquer des fichiers CSV, ou utiliser --photos")
    from viewer import apply_saved_poses   # import ici : viewer.py modifie les raccourcis matplotlib
    paths = []
    for pattern in args.files:
        paths += sorted(glob(pattern)) or [pattern]
    scans = [load_scan(p, r_max=args.rmax) for p in paths]
    apply_saved_poses(scans, args.poses)
    if not args.tous_les_tours:
        scans = [premier_tour(s) for s in scans]
    return scans, Path.cwd()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", help="fichiers CSV")
    ap.add_argument("--photos", action="store_true", help="utilise les photos de photos/ (voir photo_multi.py)")
    ap.add_argument("--poses", default="poses.json")
    ap.add_argument("--rmax", type=float, default=12.0, help="portée max conservée (m) ; 6-8 réduit le bruit lointain")
    ap.add_argument("--tous-les-tours", action="store_true",
                    help="garde tous les tours d'une photo (par défaut : le premier tour seulement)")
    ap.add_argument("--min-obs", type=int, default=1, help="ne garder que les segments fusionnés >= N fois")
    ap.add_argument("--eps", type=float, default=Params.eps, help="tolérance Douglas-Peucker (m)")
    ap.add_argument("--min-len", type=float, default=Params.min_len, help="longueur minimale d'un segment (m)")
    ap.add_argument("--tol-dist", type=float, default=Params.tol_dist, help="fusion : écart max entre droites (m)")
    ap.add_argument("--tol-angle", type=float, default=Params.tol_angle, help="fusion : angle max (degrés)")
    ap.add_argument("--tol-gap", type=float, default=Params.tol_gap, help="fusion : trou max le long de l'axe (m)")
    ap.add_argument("--no-show", action="store_true", help="n'ouvre pas la fenêtre (enregistre seulement)")
    args = ap.parse_args()

    scans, sortie = charger_scans(args)
    params = Params(eps=args.eps, min_len=args.min_len, tol_dist=args.tol_dist,
                    tol_angle=args.tol_angle, tol_gap=args.tol_gap)
    carte = CarteVectorielle(params)
    for scan in scans:
        carte.mettre_a_jour(scan, (scan.pose[0], scan.pose[1]))
        print(f"{scan.name}: {len(carte.segments)} segments dans la carte")
    carte.nettoyer(args.min_obs)

    csv_path, png_path = Path(sortie) / "carte_vectorielle.csv", Path(sortie) / "carte_vectorielle.png"
    carte.sauver(csv_path)
    print(f"{len(carte.segments)} segments -> {csv_path}")
    afficher_carte_vectorielle(carte, scans, png=png_path, montrer=not args.no_show)
    print(f"Image enregistrée : {png_path}")


if __name__ == "__main__":
    main()