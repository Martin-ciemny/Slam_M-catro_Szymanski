import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial

# --- CONFIGURATION ---
PORT_ARDUINO = "COM5"   # Votre port Arduino
BAUDRATE = 115200       # Vitesse de votre code Arduino
NOM_FICHIER = "capture_lidar.csv"
TIMEOUT_S = 15          # Abandon si aucun scan complet n'est reçu dans ce délai

print(f"🔌 Connexion à l'Arduino sur {PORT_ARDUINO}...")
try:
    ser = serial.Serial(PORT_ARDUINO, BAUDRATE, timeout=1)
    time.sleep(2)  # Pause de sécurité pour laisser l'Arduino redémarrer
    ser.reset_input_buffer()  # On vide le tampon pour avoir des données fraîches
except Exception as e:
    print(f"Impossible de se connecter : {e}")
    print("Vérifiez que le Serial Monitor d'Arduino est bien FERMÉ.")
    sys.exit()


def lire_ligne(ser):
    """Renvoie (N, angle_deg, distance_mm, texte_brut) ou None si la ligne est vide/coupée/illisible."""
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


# 1. PRISE DE LA "PHOTO" : on attend UN scan complet (de N=0 jusqu'au N=0 suivant)
print("📸 Flash ! Attente d'un scan complet...")
ser.readline()  # la première ligne reçue est souvent coupée : on la jette
scan = []
photo = None
debut = time.time()

while photo is None and time.time() - debut < TIMEOUT_S:
    ligne = lire_ligne(ser)
    if ligne is None:
        continue
    if ligne[0] == 0 and scan:        # N repart de 0 : le scan précédent est terminé
        if scan_valide(scan):
            photo = scan
        scan = []                     # sinon on l'abandonne et on repart sur le suivant
    if scan and ligne == scan[-1]:    # doublon exact (artefact série)
        continue
    scan.append(ligne)

# Fermeture immédiate du port pour libérer l'Arduino
ser.close()

if photo is None:
    sys.exit("Aucun scan complet reçu : vérifiez le port, le baudrate et que "
             "printScanCSV est bien activé dans le .ino.")
print(f"✅ Capture terminée. {len(photo)} points enregistrés.")

# 2. ENREGISTREMENT DANS LE FICHIER CSV : les lignes brutes reçues, comme avant
with open(NOM_FICHIER, "w", encoding="utf-8") as f:
    f.write("N,Angle(deg),Distance(mm),Intensity,x(mm),y(mm)\n")
    for _, _, _, texte in photo:
        f.write(texte + "\n")
print(f"💾 Fichier écrit : {NOM_FICHIER}")

# 3. AFFICHAGE DE L'IMAGE FIXE
df = pd.DataFrame([r[:3] for r in photo], columns=["N", "angle_deg", "dist_mm"])
a = np.deg2rad(df["angle_deg"])
df["X"] = df["dist_mm"] / 1000.0 * np.cos(a)
df["Y"] = df["dist_mm"] / 1000.0 * np.sin(a)   # signe corrigé : le y de la librairie est inversé

nb_zero = int((df["dist_mm"] <= 0).sum())
print(f"   distances : min={df['dist_mm'].min():.0f} mm, max={df['dist_mm'].max():.0f} mm, "
      f"{nb_zero} point(s) à 0 mm (pas d'impact)")
print("   3 premières lignes reçues :")
for _, _, _, texte in photo[:3]:
    print("   ", texte)

df = df[df["dist_mm"] > 0]  # distance 0 = pas d'impact
if df.empty:
    sys.exit("Tous les points ont une distance de 0 : le lidar ne renvoie aucune mesure "
             "(alimentation, rotation du moteur, câble de données ?).")

print("🖼️ Ouverture de la photo cartographique...")
plt.figure(figsize=(8, 8))
plt.scatter(df["X"], df["Y"], color="black", s=2, marker=".")
plt.plot(0, 0, "ro")  # position du lidar

plt.title(f"Photo LiDAR - {len(df)} points")
plt.xlabel("Axe X (m)")
plt.ylabel("Axe Y (m)")
plt.grid(True)
plt.axis("equal")  # Très important pour ne pas déformer la pièce
plt.show()
