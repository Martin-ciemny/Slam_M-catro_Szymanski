"""Cartographie et localisation en continu (Grid Map + Carte Vectorielle)."""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grid_map_new import GridMap, localisation_brute_force, mettre_a_jour_carte
from panda2 import BAUDRATE, PORT_ARDUINO, TIMEOUT_S, lire_ligne, scan_valide
from scan_io import Scan
from vector_map import CarteVectorielle, Params


def lignes_vers_scan(nom: str, rows: list[tuple[int, float, float, str]],
                     r_min: float = 0.02, r_max: float = 12.0) -> Scan | None:
    """Convertit un tour brut [(N, angle_deg, dist_mm, raw), ...] en Scan."""
    if not rows:
        return None
    angles = np.array([r[1] for r in rows], dtype=float)
    dists = np.array([r[2] for r in rows], dtype=float) / 1000.0
    keep = (dists >= r_min) & (dists <= r_max)
    if not np.any(keep):
        return None
    return Scan(name=nom, angle_deg=angles[keep], r_m=dists[keep])


def worker_serie(port: str, baudrate: int, q_scans: queue.Queue, stop_event: threading.Event):
    """Lit l'ESP32 en continu dans un thread séparé et envoie chaque tour complet dans q_scans."""
    try:
        import serial
    except ImportError:
        print("pyserial manque : python -m pip install pyserial")
        stop_event.set()
        return

    print(f"🔌 Connexion série continue sur {port} ({baudrate} bauds)...")
    try:
        ser = serial.Serial(port, baudrate, timeout=0.2)
        time.sleep(2.0)  # laisse l'ESP32 démarrer si reset à l'ouverture
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Erreur port série : {e}")
        stop_event.set()
        return

    ser.readline()  # première ligne souvent incomplète
    scan_courant = []
    num_scan = 0
    derniere_donnee = time.time()

    try:
        while not stop_event.is_set():
            ligne = lire_ligne(ser)
            if ligne is None:
                if time.time() - derniere_donnee > TIMEOUT_S:
                    print("⚠️ Aucune donnée Lidar reçue depuis plusieurs secondes...")
                    derniere_donnee = time.time()
                continue

            derniere_donnee = time.time()

            # N repart à 0 : on a fini le tour précédent
            if ligne[0] == 0 and scan_courant:
                if scan_valide(scan_courant):
                    num_scan += 1
                    s = lignes_vers_scan(f"scan_{num_scan:04d}", scan_courant)
                    if s is not None:
                        # Si la file est pleine, on jette le plus ancien pour garder du temps réel
                        if q_scans.full():
                            try:
                                q_scans.get_nowait()
                            except queue.Empty:
                                pass
                        q_scans.put(s)
                scan_courant = []

            if scan_courant and ligne == scan_courant[-1]:
                continue
            scan_courant.append(ligne)
    finally:
        ser.close()
        print("🛑 Connexion série fermée.")


def main():
    ap = argparse.ArgumentParser(description="SLAM continu simple (Grid Map + Carte Vectorielle)")
    ap.add_argument("--port", default=PORT_ARDUINO, help="Port série ESP32 (ex: COM5 ou /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=BAUDRATE, help="Vitesse série")
    ap.add_argument("--taille", type=float, default=10.0, help="Demi-taille de la carte en mètres (+/- taille)")
    ap.add_argument("--res", type=float, default=0.05, help="Résolution de la grille (m/case)")
    ap.add_argument("--pas-localisation", type=int, default=1,
                    help="Localise tous les N scans (1 = à chaque scan)")
    ap.add_argument("--sans-loc", action="store_true",
                    help="Désactive le recalage automatique (utile si tu veux tester pose fixe ou odométrie)")
    args = ap.parse_args()

    carte_grid = GridMap(
        xmin=-args.taille, xmax=args.taille,
        ymin=-args.taille, ymax=args.taille,
        resolution=args.res
    )
    carte_vec = CarteVectorielle(Params())

    q_scans: queue.Queue[Scan] = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    t_serie = threading.Thread(
        target=worker_serie,
        args=(args.port, args.baud, q_scans, stop_event),
        daemon=True
    )
    t_serie.start()

    plt.ion()
    fig, (ax_grid, ax_vec) = plt.subplots(1, 2, figsize=(13, 6))
    extent = [carte_grid.xmin, carte_grid.xmax, carte_grid.ymin, carte_grid.ymax]

    img_grid = ax_grid.imshow(
        carte_grid.grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=255
    )
    traj_x, traj_y = [], []
    ligne_traj_g, = ax_grid.plot([], [], "-b", lw=1.5, label="Trajectoire")
    pos_g, = ax_grid.plot([], [], "ro", ms=6)
    ax_grid.set_title("Grid Map en continu")
    ax_grid.set_xlabel("X (m)")
    ax_grid.set_ylabel("Y (m)")
    ax_grid.legend(loc="upper right")

    ligne_traj_v, = ax_vec.plot([], [], "-r", lw=1.2, alpha=0.7, label="Trajectoire")
    pos_v, = ax_vec.plot([], [], "ro", ms=6)
    ax_vec.set_title("Carte Vectorielle en continu")
    ax_vec.set_xlabel("X (m)")
    ax_vec.set_ylabel("Y (m)")
    ax_vec.set_aspect("equal", adjustable="datalim")
    ax_vec.grid(True, alpha=0.3)

    pose_courante = (0.0, 0.0, 0.0)
    nb_scans_traites = 0
    a_des_murs = False

    def fermer(_event=None):
        stop_event.set()

    fig.canvas.mpl_connect("close_event", fermer)

    print("🚀 SLAM continu démarré. Fermez la fenêtre graphique pour arrêter.")
    try:
        while not stop_event.is_set():
            try:
                scan = q_scans.get(timeout=0.1)
            except queue.Empty:
                fig.canvas.flush_events()
                continue

            nb_scans_traites += 1

            # 1. Localisation autour de la dernière pose connue (si la carte a déjà des murs)
            if not args.sans_loc and a_des_murs and (nb_scans_traites % max(1, args.pas_localisation) == 0):
                pose_estimee = localisation_brute_force(carte_grid, scan, pose_courante)
            else:
                pose_estimee = pose_courante

            scan.set_pose(*pose_estimee)
            pose_courante = pose_estimee

            # 2. Mise à jour des deux cartes
            mettre_a_jour_carte(carte_grid, scan)
            carte_vec.mettre_a_jour(scan, position_robot=pose_courante[:2])

            if np.any(carte_grid.grid > 180):
                a_des_murs = True

            # 3. Mise à jour de l'affichage
            traj_x.append(pose_courante[0])
            traj_y.append(pose_courante[1])

            img_grid.set_data(carte_grid.grid)
            ligne_traj_g.set_data(traj_x, traj_y)
            pos_g.set_data([pose_courante[0]], [pose_courante[1]])

            # Redessine les segments vectoriels proprement
            for artist in list(ax_vec.lines):
                if artist not in (ligne_traj_v, pos_v):
                    artist.remove()
            for seg in carte_vec.segments:
                couleur = "tab:blue" if seg.n_obs >= 2 else "tab:orange"
                epaisseur = 2.5 if seg.n_obs >= 2 else 1.5
                ax_vec.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]], "-", color=couleur, lw=epaisseur)

            ligne_traj_v.set_data(traj_x, traj_y)
            pos_v.set_data([pose_courante[0]], [pose_courante[1]])
            ax_vec.relim()
            ax_vec.autoscale_view()

            ax_grid.set_title(
                f"Grid Map | Scan #{nb_scans_traites} | "
                f"x={pose_courante[0]:.2f}m y={pose_courante[1]:.2f}m th={pose_courante[2]:.0f}°"
            )
            ax_vec.set_title(f"Carte Vectorielle ({len(carte_vec.segments)} segments)")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        t_serie.join(timeout=1.0)
        out_dir = Path(__file__).parent / "photos"
        out_dir.mkdir(exist_ok=True)
        carte_vec.sauver(out_dir / "carte_vectorielle_continue.csv")
        fig.savefig(out_dir / "slam_continu.png", dpi=150)
        print(f"💾 Sauvegardé dans {out_dir}")
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()