"""Outil pour capturer des scans Lidar (photos) et renseigner les déplacements du robot."""
import argparse
import csv
import re
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration matérielle et chemins
PORT_ARDUINO = "COM5"
BAUDRATE = 115200
TIMEOUT_S = 15
DOSSIER = Path(__file__).parent / "photos"
FICHIER_DEPL = DOSSIER / "deplacements.csv"
COLONNES = ["N", "Angle(deg)", "Distance(mm)", "Intensity", "x(mm)", "y(mm)"]

def capturer_photo() -> list:
    """Se connecte à l'ESP32 et écoute le port série pour récupérer un tour complet du Lidar."""
    import serial
    print(f"Connexion à {PORT_ARDUINO}...")
    ser = serial.Serial(PORT_ARDUINO, BAUDRATE, timeout=1)
    time.sleep(4) # Laisse le temps à l'ESP32 de redémarrer
    ser.reset_input_buffer()
    
    print("Enregistrement du scan... Ne bougez pas le robot.")
    ser.readline() # On ignore la première ligne souvent tronquée
    
    scan, photo_complete = [], None
    debut = time.time()
    
    while photo_complete is None and time.time() - debut < TIMEOUT_S:
        raw = ser.readline().decode("utf-8", errors="replace").strip()
        parts = raw.split(",")
        if len(parts) < 4:
            continue
            
        try:
            n = int(parts[0])
            # Si N repart à 0 et qu'on a déjà lu plus de 50 points, on a un tour complet
            if n == 0 and len(scan) >= 50:
                photo_complete = scan
                break
            scan.append(raw)
        except ValueError:
            pass
            
    ser.close()
    if not photo_complete:
        sys.exit("Erreur : Aucun scan complet reçu dans le temps imparti.")
    return photo_complete

def lire_deplacements() -> dict:
    """Lit le fichier CSV des déplacements et retourne un dictionnaire {photo: (dx, dy, dtheta)}."""
    if not FICHIER_DEPL.exists():
        return {}
    
    deplacements = {}
    lignes = FICHIER_DEPL.read_text(encoding="utf-8").splitlines()
    for ligne in csv.reader(lignes):
        if not ligne or ligne[0] == "photo":
            continue # Ignore l'en-tête ou les lignes vides
            
        nom = ligne[0].strip()
        try:
            # On utilise une compréhension de liste pour convertir les 3 valeurs en float
            dx, dy, dth = [float(v.replace(",", ".")) for v in ligne[1:4]]
            deplacements[nom] = (dx, dy, dth)
        except (ValueError, IndexError):
            deplacements[nom] = None # Déplacement manquant ou mal formaté
            
    return deplacements

def calculer_poses(noms_photos: list, deplacements: dict) -> dict:
    """Calcule la position absolue (pose) de chaque photo en cumulant les déplacements relatifs."""
    poses = {}
    x = y = th = 0.0 # La première photo est toujours l'origine (0, 0, 0)
    
    for nom in noms_photos:
        dep = deplacements.get(nom, (0.0, 0.0, 0.0))
        if dep is None:
            dep = (0.0, 0.0, 0.0)
            
        x += dep[0]
        y += dep[1]
        th += dep[2]
        poses[nom] = (x, y, th)
        
    return poses

def main():
    # Création du dossier si inexistant
    DOSSIER.mkdir(exist_ok=True)
    
    # 1. Capture d'une nouvelle photo Lidar
    photo = capturer_photo()
    index_photo = len(list(DOSSIER.glob("photo_*.csv"))) + 1
    nom_photo = f"photo_{index_photo:03d}"
    
    # 2. Sauvegarde du fichier CSV du scan
    chemin_csv = DOSSIER / f"{nom_photo}.csv"
    with open(chemin_csv, "w", encoding="utf-8") as f:
        f.write(",".join(COLONNES) + "\n")
        f.write("\n".join(photo))
    print(f"Photo sauvegardée : {nom_photo}.csv")
    
    # 3. Demande du déplacement à l'utilisateur
    deplacements = lire_deplacements()
    if index_photo == 1:
        deplacements[nom_photo] = (0.0, 0.0, 0.0)
    else:
        saisie = input("Déplacement depuis la dernière photo ? format: dx dy dtheta (ex: 1.5 0.0 -90) : ")
        # L'expression régulière 're.split' sépare la saisie par les espaces ou points-virgules
        vals = [float(v) for v in re.split(r"[;\s]+", saisie.strip()) if v]
        deplacements[nom_photo] = tuple(vals) if len(vals) == 3 else (0.0, 0.0, 0.0)
        
    # Mise à jour du fichier CSV des déplacements
    with open(FICHIER_DEPL, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["photo", "dx_m", "dy_m", "dtheta_deg"])
        for nom, dep in deplacements.items():
            writer.writerow([nom, dep[0], dep[1], dep[2]] if dep else [nom, "", "", ""])

if __name__ == "__main__":
    main()