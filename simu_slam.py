"""Simulation SLAM en continu dans une pièce en L avec des obstacles."""
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from grid_map_new import GridMap, localisation_brute_force, mettre_a_jour_carte
from scan_io import Scan, wrap_deg
from vector_map import CarteVectorielle, Params


def construire_piece_en_L() -> list[tuple[float, float, float, float]]:
    """Renvoie la liste des segments (x1, y1, x2, y2) : une pièce en L + 2 obstacles."""
    murs = [
        # Contour extérieur en forme de L
        (-3.0, -2.0, 3.0, -2.0),   # mur du bas
        (3.0, -2.0, 3.0, 3.0),     # mur droit (branche verticale)
        (3.0, 3.0, 0.5, 3.0),      # mur du haut de la branche verticale
        (0.5, 3.0, 0.5, 0.5),      # paroi verticale intérieure (forme le L)
        (0.5, 0.5, -3.0, 0.5),     # paroi horizontale intérieure (forme le L)
        (-3.0, 0.5, -3.0, -2.0),   # mur gauche
    ]

    obstacles = [
        # Obstacle 1 : bloc carré (0.6m x 0.6m) dans la partie basse
        (-1.2, -1.2, -0.6, -1.2),
        (-0.6, -1.2, -0.6, -0.6),
        (-0.6, -0.6, -1.2, -0.6),
        (-1.2, -0.6, -1.2, -1.2),

        # Obstacle 2 : pilier rectangulaire (0.6m x 0.8m) dans la branche verticale
        (1.4, 0.8, 2.0, 0.8),
        (2.0, 0.8, 2.0, 1.6),
        (2.0, 1.6, 1.4, 1.6),
        (1.4, 1.6, 1.4, 0.8),
    ]

    return murs + obstacles


def distance_rayon_segment(
    xr: float, yr: float, dx: float, dy: float,
    x1: float, y1: float, x2: float, y2: float
) -> float | None:
    """Calcule la distance t > 0 entre le rayon (xr, yr) + t*(dx, dy) et le segment [A, B]."""
    sx = x2 - x1
    sy = y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None

    qpx = x1 - xr
    qpy = y1 - yr

    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * dy - qpy * dx) / denom

    if t > 1e-4 and 0.0 <= u <= 1.0:
        return t
    return None


def simuler_scan_piece(
    nom: str,
    pose_vraie: tuple[float, float, float],
    murs: list[tuple[float, float, float, float]],
    pas_deg: float = 2.0,
    bruit_m: float = 0.015,
    r_max: float = 10.0,
) -> Scan:
    """Simule un tour Lidar complet face à la liste de segments `murs`."""
    xr, yr, th_deg = pose_vraie
    angles_locaux = np.arange(0.0, 360.0, pas_deg)
    distances = []

    for ang_loc in angles_locaux:
        ang_monde = math.radians(wrap_deg(th_deg + ang_loc))
        dx = math.cos(ang_monde)
        dy = math.sin(ang_monde)

        dists_touchees = []
        for x1, y1, x2, y2 in murs:
            d = distance_rayon_segment(xr, yr, dx, dy, x1, y1, x2, y2)
            if d is not None:
                dists_touchees.append(d)

        d_min = min(dists_touchees) if dists_touchees else r_max
        d_bruitee = d_min + np.random.normal(0.0, bruit_m)

        if 0.05 <= d_bruitee <= r_max:
            distances.append(d_bruitee)
        else:
            distances.append(0.0)

    ang_arr = np.array(angles_locaux, dtype=float)
    dist_arr = np.array(distances, dtype=float)
    valides = dist_arr > 0
    return Scan(name=nom, angle_deg=ang_arr[valides], r_m=dist_arr[valides])


def construire_trajectoire_L() -> list[tuple[float, float, float]]:
    """Trajectoire simple : avance vers la droite puis monte dans la branche verticale du L."""
    poses = []
    for i in range(18):
        poses.append((-2.0 + i * 0.20, -0.2, 0.0))

    for i in range(1, 7):
        poses.append((1.6, -0.2, i * 15.0))

    for i in range(1, 13):
        poses.append((1.6, -0.2 + i * 0.20, 90.0))

    return poses


def main():
    np.random.seed(42)
    murs = construire_piece_en_L()
    poses_vraies = construire_trajectoire_L()

    carte_grid = GridMap(xmin=-5.0, xmax=5.0, ymin=-4.0, ymax=5.0, resolution=0.05)
    params_vec = Params(min_len=0.30)
    carte_vec = CarteVectorielle(params_vec)

    plt.ion()
    fig, (ax_grid, ax_vec) = plt.subplots(1, 2, figsize=(13, 6))
    extent = [carte_grid.xmin, carte_grid.xmax, carte_grid.ymin, carte_grid.ymax]

    img_grid = ax_grid.imshow(
        carte_grid.grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=255
    )
    traj_x, traj_y = [], []
    ligne_traj_g, = ax_grid.plot([], [], "-b", lw=1.5, label="Trajectoire estimée")
    pos_g, = ax_grid.plot([], [], "ro", ms=6)
    ax_grid.set_title("Grid Map (Pièce en L + obstacles)")
    ax_grid.set_xlabel("X (m)")
    ax_grid.set_ylabel("Y (m)")
    ax_grid.legend(loc="upper right")

    ligne_traj_v, = ax_vec.plot([], [], "-r", lw=1.2, alpha=0.7, label="Trajectoire estimée")
    pos_v, = ax_vec.plot([], [], "ro", ms=6)
    ax_vec.set_title("Carte Vectorielle (Pièce en L + obstacles)")
    ax_vec.set_xlabel("X (m)")
    ax_vec.set_ylabel("Y (m)")
    ax_vec.set_aspect("equal", adjustable="datalim")
    ax_vec.grid(True, alpha=0.3)

    pose_estimee = poses_vraies[0]
    a_des_murs = False

    for idx, pose_vraie in enumerate(poses_vraies):
        scan = simuler_scan_piece(f"simu_{idx:03d}", pose_vraie, murs)

        if idx == 0:
            pose_depart = pose_vraie
        else:
            dx_prev = poses_vraies[idx][0] - poses_vraies[idx - 1][0]
            dy_prev = poses_vraies[idx][1] - poses_vraies[idx - 1][1]
            dth_prev = poses_vraies[idx][2] - poses_vraies[idx - 1][2]
            pose_depart = (
                pose_estimee[0] + dx_prev + np.random.normal(0.0, 0.03),
                pose_estimee[1] + dy_prev + np.random.normal(0.0, 0.03),
                pose_estimee[2] + dth_prev + np.random.normal(0.0, 1.5),
            )

        if a_des_murs:
            pose_estimee = localisation_brute_force(carte_grid, scan, pose_depart)
        else:
            pose_estimee = pose_depart

        scan.set_pose(*pose_estimee)
        mettre_a_jour_carte(carte_grid, scan)
        carte_vec.mettre_a_jour(scan, position_robot=pose_estimee[:2])

        if np.any(carte_grid.grid > 180):
            a_des_murs = True

        traj_x.append(pose_estimee[0])
        traj_y.append(pose_estimee[1])

        img_grid.set_data(carte_grid.grid)
        ligne_traj_g.set_data(traj_x, traj_y)
        pos_g.set_data([pose_estimee[0]], [pose_estimee[1]])

        for artist in list(ax_vec.lines):
            if artist not in (ligne_traj_v, pos_v):
                artist.remove()
        for seg in carte_vec.segments:
            couleur = "tab:blue" if seg.n_obs >= 2 else "tab:orange"
            epaisseur = 2.5 if seg.n_obs >= 2 else 1.5
            ax_vec.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], "-", color=couleur, lw=epaisseur)

        ligne_traj_v.set_data(traj_x, traj_y)
        pos_v.set_data([pose_estimee[0]], [pose_estimee[1]])
        ax_vec.relim()
        ax_vec.autoscale_view()

        err_cm = math.hypot(pose_estimee[0] - pose_vraie[0], pose_estimee[1] - pose_vraie[1]) * 100.0
        ax_grid.set_title(
            f"Scan #{idx + 1}/{len(poses_vraies)} | "
            f"x={pose_estimee[0]:.2f}m y={pose_estimee[1]:.2f}m th={pose_estimee[2]:.0f}° | err={err_cm:.1f}cm"
        )
        ax_vec.set_title(f"Carte Vectorielle ({len(carte_vec.segments)} segments)")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.12)

    print("✅ Simulation terminée. Fermez la fenêtre pour quitter.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()