"""Photos lidar à plusieurs endroits de la salle, tracées sur une même carte.

Usage :
    python photo_multi.py            prend une nouvelle photo, demande le déplacement, retrace la carte
    python photo_multi.py --show     retrace la carte sans nouvelle photo (après avoir édité les déplacements)

Principe :
  - Chaque exécution enregistre photos/photo_001.csv, photo_002.csv, ... (lignes brutes de l'ESP32).
  - La photo 1 est l'ORIGINE de la carte : pose (0, 0, 0). Ses axes x, y définissent ceux de la carte.
  - Pour les photos suivantes, on indique le DÉPLACEMENT depuis la photo précédente :
        dx (m), dy (m)   dans les axes de la carte (ceux de la photo 1)
        dtheta (deg)     rotation du lidar, positive vers la gauche (sens antihoraire)
    Les poses sont obtenues en additionnant ces déplacements.
  - Les déplacements sont dans photos/deplacements.csv : on peut les saisir au moment de la photo
    (ou appuyer sur Entrée pour les saisir plus tard), puis les modifier à tout moment avec un éditeur
    de texte (VS Code) et lancer  python photo_multi.py --show.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
PORT_ARDUINO = "COM5"   # Port de l'ESP32
BAUDRATE = 115200       # Même valeur que Serial.begin() dans le .ino
TIMEOUT_S = 15          # Abandon si aucun scan complet n'est reçu dans ce délai
DOSSIER = Path(__file__).parent / "photos"
FICHIER_DEPL = DOSSIER / "deplacements.csv"
COLONNES = ["N", "Angle(deg)", "Distance(mm)", "Intensity", "x(mm)", "y(mm)"]
LONGUEUR_FLECHE = 0.25  # m, orientation du lidar sur la carte


# ----------------------------------------------------------------- capture
def lire_ligne(ser):
    """Renvoie (N, angle_deg, distance_mm, texte_brut) ou None si vide/coupée/illisible."""
    raw = ser.readline()
    if not raw:
        return None
    texte = raw.decode("utf-8", errors="replace").strip()
    parts = texte.split(",")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), texte
    except ValueError:
        return None


def scan_valide(rows):
    """Valide si N vaut 0, 1, 2, ... sans trou (sinon une ligne a été perdue ou coupée)."""
    return len(rows) >= 50 and [r[0] for r in rows] == list(range(len(rows)))


def capturer_photo():
    """Attend UN scan complet sur le port série et le renvoie (liste de lignes)."""
    try:
        import serial
    except ImportError:
        sys.exit("pyserial manque :  python -m pip install pyserial")

    print(f"🔌 Connexion à l'ESP32 sur {PORT_ARDUINO}...")
    try:
        ser = serial.Serial(PORT_ARDUINO, BAUDRATE, timeout=1)
        time.sleep(4)  # l'ESP32 redémarre à l'ouverture du port : on le laisse démarrer
        ser.reset_input_buffer()
    except Exception as e:
        sys.exit(f"Impossible de se connecter : {e}\nVérifiez que le Moniteur série d'Arduino est FERMÉ.")

    print("📸 Flash ! Attente d'un scan complet (ne bougez pas le robot)...")
    ser.readline()  # la première ligne reçue est souvent coupée : on la jette
    scan, photo = [], None
    debut = time.time()
    while photo is None and time.time() - debut < TIMEOUT_S:
        ligne = lire_ligne(ser)
        if ligne is None:
            continue
        if ligne[0] == 0 and scan:        # N repart de 0 : le scan précédent est terminé
            if scan_valide(scan):
                photo = scan
            scan = []
        if scan and ligne == scan[-1]:    # doublon exact (artefact série)
            continue
        scan.append(ligne)
    ser.close()

    if photo is None:
        sys.exit("Aucun scan complet reçu : vérifiez le port, le baudrate et que "
                 "printScanCSV est activé dans le .ino.")
    return photo


# ----------------------------------------------------------------- fichiers
def liste_photos():
    return sorted(DOSSIER.glob("photo_[0-9][0-9][0-9].csv"))


def nom_prochaine_photo():
    nums = [int(re.search(r"(\d+)", p.stem).group(1)) for p in liste_photos()]
    return f"photo_{(max(nums) + 1 if nums else 1):03d}"


def sauver_photo(nom, photo):
    chemin = DOSSIER / f"{nom}.csv"
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(",".join(COLONNES) + "\n")
        for _, _, _, texte in photo:
            f.write(texte + "\n")
    return chemin


def _nombre(texte):
    """'0,25' ou '0.25' -> 0.25 ; vide -> None."""
    texte = texte.strip().replace(",", ".")
    return float(texte) if texte else None


def lire_deplacements():
    """{nom_photo: (dx, dy, dtheta) ou None si non renseigné}. Tolère ';' et virgules décimales."""
    if not FICHIER_DEPL.exists():
        return {}
    lignes = FICHIER_DEPL.read_text(encoding="utf-8-sig").splitlines()
    delim = ";" if lignes and ";" in lignes[0] else ","
    depl = {}
    for ligne in list(csv.reader(lignes, delimiter=delim))[1:]:
        if not ligne or not ligne[0].strip():
            continue
        try:
            vals = [_nombre(v) for v in (ligne[1:4] + ["", "", ""])[:3]]
        except ValueError:
            vals = [None, None, None]
        depl[ligne[0].strip()] = None if None in vals else tuple(vals)
    return depl


def ecrire_deplacements(depl):
    with open(FICHIER_DEPL, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "dx_m", "dy_m", "dtheta_deg"])
        for nom in sorted(depl):
            d = depl[nom]
            w.writerow([nom] + (["", "", ""] if d is None else list(d)))


def demander_deplacement():
    """Saisie : 'dx dy dtheta' (espaces ou ;). Entrée seule = à renseigner plus tard."""
    print("Déplacement depuis la photo précédente (axes de la carte, mètres et degrés)")
    while True:
        rep = input("  dx dy dtheta  (ex: 1.5 0.3 -20, Entrée = plus tard) : ").strip()
        if not rep:
            return None
        try:
            vals = [float(v.replace(",", ".")) for v in re.split(r"[;\s]+", rep)]
            if len(vals) == 3:
                return tuple(vals)
        except ValueError:
            pass
        print("  -> il faut trois nombres séparés par des espaces, par exemple : 1.5 0.3 -20")


# ----------------------------------------------------------------- carte
def calculer_poses(noms, depl):
    """Cumule les déplacements : pose_k = pose_(k-1) + déplacement_k. Photo 1 = origine."""
    poses = {}
    x = y = th = 0.0
    for k, nom in enumerate(noms):
        if k > 0:
            d = depl.get(nom)
            if d is None:
                print(f"⚠️  {nom} : déplacement non renseigné, supposé nul "
                      f"(à compléter dans {FICHIER_DEPL.name} puis --show)")
                d = (0.0, 0.0, 0.0)
            x, y, th = x + d[0], y + d[1], th + d[2]
        poses[nom] = (x, y, th)
    return poses


def points_monde(chemin, pose):
    """Points (N,2) d'une photo dans le repère de la carte."""
    df = pd.read_csv(chemin)
    angle, dist = df.iloc[:, 1].to_numpy(float), df.iloc[:, 2].to_numpy(float) / 1000.0
    ok = dist > 0                                   # distance 0 = pas d'impact
    x0, y0, th = pose
    a = np.deg2rad(angle[ok] + th)
    return np.column_stack((x0 + dist[ok] * np.cos(a), y0 + dist[ok] * np.sin(a)))


def tracer_carte():
    chemins = liste_photos()
    if not chemins:
        sys.exit("Aucune photo dans le dossier photos/.")
    noms = [c.stem for c in chemins]
    depl = lire_deplacements()
    poses = calculer_poses(noms, depl)
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(9, 9))
    for i, (nom, chemin) in enumerate(zip(noms, chemins)):
        pts = points_monde(chemin, poses[nom])
        x0, y0, th = poses[nom]
        c = cmap(i % 10)
        plt.scatter(pts[:, 0], pts[:, 1], s=3, color=c,
                    label=f"{nom}  (x={x0:.2f} m, y={y0:.2f} m, θ={th:.0f}°, {len(pts)} pts)")
        plt.plot(x0, y0, "o", color=c, markeredgecolor="k", ms=8)
        t = np.deg2rad(th)
        plt.plot([x0, x0 + LONGUEUR_FLECHE * np.cos(t)], [y0, y0 + LONGUEUR_FLECHE * np.sin(t)],
                 "-", color=c, lw=2)

    plt.title(f"Carte lidar - {len(noms)} photo(s)")
    plt.xlabel("Axe X (m)")
    plt.ylabel("Axe Y (m)")
    plt.grid(True, alpha=0.4)
    plt.axis("equal")
    plt.legend(loc="best", fontsize=8, markerscale=3)
    plt.savefig(DOSSIER / "carte.png", dpi=150)
    print(f"🖼️  Carte enregistrée : {DOSSIER / 'carte.png'}")
    plt.show()


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--show", action="store_true", help="retrace la carte sans nouvelle photo")
    args = ap.parse_args()
    DOSSIER.mkdir(exist_ok=True)

    if not args.show:
        nom = nom_prochaine_photo()
        photo = capturer_photo()
        chemin = sauver_photo(nom, photo)
        print(f"✅ {len(photo)} points enregistrés dans {chemin}")

        depl = lire_deplacements()
        if nom == "photo_001":
            print("Photo 1 = origine de la carte : pose (0, 0, 0).")
            depl[nom] = (0.0, 0.0, 0.0)
        else:
            depl[nom] = demander_deplacement()
        ecrire_deplacements(depl)

    tracer_carte()


if __name__ == "__main__":
    main()
