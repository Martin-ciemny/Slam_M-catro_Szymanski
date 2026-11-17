"""TP1 - étape 2 : visualiseur de nuages de points lidar.

Usage :
    python viewer.py --photos                    calibre les photos de photos/ (voir panda2.py)
    python viewer.py scan1.csv scan2.csv ...
    python viewer.py scans/*.csv --poses poses.json
    python viewer.py scan1.csv --lidar-cw        (angles bruts du LD06, sens horaire)

Chaque scan a sa propre couleur et sa propre pose (x, y, theta) que l'on peut
modifier au clavier pour recaler les scans les uns sur les autres.
"""
import argparse
import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scan_io import Scan, load_photo_csv, load_scan, wrap_deg

# On libère les raccourcis par défaut de matplotlib (s, q, flèches, r, ...)
for _k in list(plt.rcParams):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = []

HELP = (
    "Tab / Maj+Tab ou 1-9 : choisir le scan   |  flèches : déplacer (Maj x5, Ctrl x0.2)\n"
    "a / e : tourner +/- (Maj x5)  |  v : masquer/afficher  |  r : pose initiale\n"
    "Ctrl : pas fin  |  f : cadrer  |  s : sauver les poses  |  h : aide"
)


class Viewer:
    STEP_XY = 0.05   # m
    STEP_TH = 2.0    # degrés
    HEAD_LEN = 0.25  # m, longueur de la flèche d'orientation

    def __init__(self, scans: list[Scan], poses_path: str = "poses.json", mode_photos: bool = False):
        self.scans = scans
        self.poses_path = poses_path
        self.mode_photos = mode_photos
        self.locked = {0} if mode_photos else set()   # photo 1 = origine de la carte, fixe
        self.sel = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("Nuages de points lidar")

        cmap = plt.get_cmap("tab10")
        self.artists = []
        for i, s in enumerate(scans):
            c = cmap(i % 10)
            cloud = self.ax.scatter([], [], s=4, color=c)
            origin, = self.ax.plot([], [], "o", color=c, markeredgecolor="k", ms=8, ls="")
            head, = self.ax.plot([], [], "-", color=c, lw=2)
            label = self.ax.text(0.01, 0.99 - 0.04 * i, "", transform=self.ax.transAxes,
                                 va="top", family="monospace", fontsize=9, color=c, zorder=10,
                                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            self.artists.append((cloud, origin, head, label))

        self.help_txt = self.ax.text(0.01, 0.01, HELP, transform=self.ax.transAxes,
                                     va="bottom", fontsize=8, family="monospace",
                                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()
        self.fit()

    # ------------------------------------------------------------------ affichage
    def update(self):
        for i, (s, (cloud, origin, head, label)) in enumerate(zip(self.scans, self.artists)):
            selected = i == self.sel
            x, y, th = s.pose
            cloud.set_offsets(s.points_world())
            cloud.set_sizes([12 if selected else 4])
            cloud.set_zorder(3 if selected else 2)
            origin.set_data([x], [y])
            t = np.deg2rad(th)
            head.set_data([x, x + self.HEAD_LEN * np.cos(t)], [y, y + self.HEAD_LEN * np.sin(t)])
            for a in (cloud, origin, head):
                a.set_visible(s.visible)
            label.set_text(f"{'>' if selected else ' '}{i + 1} {s.name[:18]:<18} "
                           f"x={x:+.3f} y={y:+.3f} th={th:+7.2f}°" + ("" if s.visible else " (masqué)")
                           + (" [origine fixe]" if i in self.locked else ""))
            label.set_fontweight("bold" if selected else "normal")
        self.fig.canvas.draw_idle()

    def fit(self):
        pts = [s.points_world() for s in self.scans if s.visible and len(s.r_m)]
        pts += [s.pose[None, :2] for s in self.scans if s.visible]
        if not pts:
            return
        p = np.vstack(pts)
        (x0, y0), (x1, y1) = p.min(axis=0) - 0.5, p.max(axis=0) + 0.5
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ clavier
    def on_key(self, event):
        k = event.key
        if k is None:
            return
        mult = 1.0
        if k.startswith("shift+") and k != "shift+tab":
            mult, k = 5.0, k[6:]
        elif k.startswith("ctrl+"):
            mult, k = 0.2, k[5:]
        elif len(k) == 1 and k.isupper():
            mult, k = 5.0, k.lower()

        s = self.scans[self.sel]
        x, y, th = s.pose
        dxy, dth = self.STEP_XY * mult, self.STEP_TH * mult

        if k == "tab":
            self.sel = (self.sel + 1) % len(self.scans)
        elif k == "shift+tab":
            self.sel = (self.sel - 1) % len(self.scans)
        elif k.isdigit() and 1 <= int(k) <= len(self.scans):
            self.sel = int(k) - 1
        elif k in ("left", "right", "up", "down", "a", "e", "r") and self.sel in self.locked:
            print("Photo 1 = origine de la carte : elle ne peut pas être déplacée.")
            return
        elif k == "left":
            s.set_pose(x - dxy, y, th)
        elif k == "right":
            s.set_pose(x + dxy, y, th)
        elif k == "up":
            s.set_pose(x, y + dxy, th)
        elif k == "down":
            s.set_pose(x, y - dxy, th)
        elif k == "a":
            s.set_pose(x, y, th + dth)
        elif k == "e":
            s.set_pose(x, y, th - dth)
        elif k == "v":
            s.visible = not s.visible
        elif k == "r":
            s.reset_pose()
        elif k == "f":
            self.fit()
        elif k == "s":
            self.save_poses()
        elif k == "h":
            self.help_txt.set_visible(not self.help_txt.get_visible())
        else:
            return
        self.update()

    # ------------------------------------------------------------------ poses
    def save_poses(self):
        if self.mode_photos:
            return self.save_deplacements()
        data = {s.name: {"x": round(float(s.pose[0]), 4),
                         "y": round(float(s.pose[1]), 4),
                         "theta_deg": round(float(s.pose[2]), 3)} for s in self.scans}
        Path(self.poses_path).write_text(json.dumps(data, indent=2))
        print(f"Poses sauvegardées dans {Path(self.poses_path).absolute()}")


    def save_deplacements(self):
        """Écrit les poses actuelles sous forme de déplacements dans photos/deplacements.csv."""
        import shutil
        import panda2 as pm
        if pm.FICHIER_DEPL.exists():
            shutil.copy(pm.FICHIER_DEPL, pm.FICHIER_DEPL.with_suffix(".bak.csv"))  # sauvegarde de sécurité
        depl, prev = {}, None
        for s in self.scans:
            if prev is None:
                depl[s.name] = (0.0, 0.0, 0.0)
            else:
                depl[s.name] = (round(float(s.pose[0] - prev[0]), 4),
                                round(float(s.pose[1] - prev[1]), 4),
                                round(float(wrap_deg(s.pose[2] - prev[2])), 3))
            prev = s.pose
        pm.ecrire_deplacements(depl)
        self.fig.savefig(pm.DOSSIER / "carte.png", dpi=150)
        print(f"Calibration enregistrée dans {pm.FICHIER_DEPL} (ancienne version : deplacements.bak.csv)")
        print(f"Carte enregistrée dans {pm.DOSSIER / 'carte.png'}")


def apply_saved_poses(scans, path):
    p = Path(path)
    if not p.exists():
        return
    data = json.loads(p.read_text())
    for s in scans:
        if s.name in data:
            d = data[s.name]
            s.set_pose(d["x"], d["y"], d["theta_deg"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", help="fichiers CSV de scans (jokers * acceptés)")
    ap.add_argument("--photos", action="store_true",
                    help="charge les photos de photos/ avec leurs déplacements ; 's' met à jour deplacements.csv")
    ap.add_argument("--lidar-cw", action="store_true",
                    help="les angles du CSV sont ceux du LD06 (sens horaire) : les inverser")
    ap.add_argument("--poses", default="poses.json", help="fichier JSON des poses (lu s'il existe, écrit avec 's')")
    ap.add_argument("--rmax", type=float, default=12.0, help="portée max conservée (m)")
    args = ap.parse_args()

    if args.photos:
        import panda2 as pm
        chemins = pm.liste_photos()
        if not chemins:
            ap.error(f"aucune photo dans {pm.DOSSIER}")
        scans = [load_photo_csv(c, r_max=args.rmax) for c in chemins]
        poses = pm.calculer_poses([s.name for s in scans], pm.lire_deplacements())
        for s in scans:
            s.set_pose(*poses[s.name])
            s.pose_init = s.pose.copy()
        for s in scans:
            print(f"{s.name}: {len(s.r_m)} points, pose = {s.pose}")
        viewer = Viewer(scans, mode_photos=True)  # à garder : matplotlib ne retient le clavier que par référence faible
        plt.show()
        return
    if not args.files:
        ap.error("indiquer des fichiers CSV, ou utiliser --photos")

    paths = []
    for pattern in args.files:          # PowerShell/cmd n'expandent pas les jokers
        paths += sorted(glob(pattern)) or [pattern]

    scans = [load_scan(p, lidar_cw=args.lidar_cw, r_max=args.rmax) for p in paths]
    apply_saved_poses(scans, args.poses)
    for s in scans:
        print(f"{s.name}: {len(s.r_m)} points, pose = {s.pose}")

    viewer = Viewer(scans, args.poses)  # à garder : matplotlib ne retient le clavier que par référence faible
    plt.show()


if __name__ == "__main__":
    main()