"""Lecture des scans lidar (CSV) et passage repère capteur -> repère monde.

Conventions (identiques à celles des outils du module) :
  - distances en mètres, angles en degrés
  - axe X vers la droite, axe Y vers le haut
  - angles positifs dans le sens ANTIHORAIRE, 0° vers +X
  - pose d'un scan = (x, y, theta_deg) du lidar dans le monde

Formats CSV acceptés (les 2 premières colonnes suffisent) :
  # Robot position: x=0.1000m, y=0.2000m, theta=30.00deg     <- optionnel
  angle_deg,r_m,x_rel_m,y_rel_m          (MapTool)
  angle_deg,distance_m,intensity         (RobotControl)

Les angles du fichier sont supposés DÉJÀ en sens antihoraire (le décodeur LD06
fait le « 360 - angle »). Si ton extraction écrit l'angle brut du LD06
(sens horaire), utilise lidar_cw=True.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_POSE_RE = re.compile(
    r"x=\s*([-+0-9.eE]+)\s*m?\s*,\s*y=\s*([-+0-9.eE]+)\s*m?\s*,\s*theta=\s*([-+0-9.eE]+)"
)


def wrap_deg(a):
    """Ramène un angle dans ]-180, 180]."""
    return 180.0 - ((180.0 - a) % 360.0)


@dataclass
class Scan:
    name: str
    angle_deg: np.ndarray            # angle local de chaque point (antihoraire)
    r_m: np.ndarray                  # distance de chaque point
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x, y, theta_deg
    pose_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    visible: bool = True

    def points_local(self) -> np.ndarray:
        """Nuage de points (N,2) dans le repère du lidar."""
        a = np.deg2rad(self.angle_deg)
        return np.column_stack((self.r_m * np.cos(a), self.r_m * np.sin(a)))

    def points_world(self) -> np.ndarray:
        """Nuage de points (N,2) dans le repère monde, selon self.pose."""
        x, y, th = self.pose
        a = np.deg2rad(self.angle_deg + th)
        return np.column_stack((x + self.r_m * np.cos(a), y + self.r_m * np.sin(a)))

    def set_pose(self, x, y, theta_deg):
        self.pose = np.array([x, y, wrap_deg(theta_deg)], dtype=float)

    def reset_pose(self):
        self.pose = self.pose_init.copy()


def load_scan(path, lidar_cw=False, r_min=0.02, r_max=12.0) -> Scan:
    """Charge un scan CSV. Écarte les points sans impact (r=0) et hors portée."""
    path = Path(path)
    pose = np.zeros(3)
    angles, dists = [], []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _POSE_RE.search(line)
            if m:
                pose = np.array([float(g) for g in m.groups()])
            continue
        parts = line.split(",")
        try:
            a, r = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue  # ligne d'en-tête ou ligne illisible
        angles.append(a)
        dists.append(r)

    if not angles:
        raise ValueError(f"{path}: aucun point lisible")

    a = np.array(angles)
    r = np.array(dists)
    if lidar_cw:
        a = (360.0 - a) % 360.0
    keep = (r >= r_min) & (r <= r_max)      # r=0 => pas d'impact, pas un obstacle
    pose[2] = wrap_deg(pose[2])
    return Scan(path.stem, a[keep], r[keep], pose.copy(), pose.copy())


def load_photo_csv(path, r_min=0.02, r_max=12.0) -> Scan:
    """Charge une photo brute de l'ESP32 : 'N,angle_deg,distance_mm,intensité,x,y' (en-tête ignoré).

    Les angles sont déjà antihoraires (librairie LD06) ; les colonnes x,y du fichier sont ignorées
    (leur signe y est inversé), on recalcule tout depuis l'angle et la distance.
    """
    path = Path(path)
    angles, dists = [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            angles.append(float(parts[1]))
            dists.append(float(parts[2]) / 1000.0)
        except ValueError:
            continue  # ligne d'en-tête
    if not angles:
        raise ValueError(f"{path}: aucun point lisible")
    a, r = np.array(angles), np.array(dists)
    keep = (r >= r_min) & (r <= r_max)
    return Scan(path.stem, a[keep], r[keep])
