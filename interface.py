"""Interface graphique (GUI) pour contrôler tous les scripts du projet Lidar."""
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# --- CHEMINS DU PROJET ---
DOSSIER_COURANT = Path(__file__).parent
DOSSIER_PHOTOS = DOSSIER_COURANT / "photos"


def compter_photos() -> list[str]:
    """Retourne la liste des fichiers CSV présents dans le dossier photos/."""
    if not DOSSIER_PHOTOS.exists():
        return []
    return [str(p) for p in sorted(DOSSIER_PHOTOS.glob("photo_*.csv"))]


def actualiser_compteur():
    """Met à jour le texte indiquant combien de scans sont disponibles."""
    fichiers = compter_photos()
    lbl_statut.config(text=f"Scans disponibles dans photos/ : {len(fichiers)}")


def lancer_commande(script_nom: str, arguments: list[str], message_console: str):
    """Lance un de nos scripts Python situés dans le dossier du projet via son chemin complet."""
    chemin_script = DOSSIER_COURANT / script_nom
    commande = [sys.executable, str(chemin_script)] + arguments
    
    print(f"\n▶ {message_console}")
    print(f"Commande exécutée : {' '.join(commande)}")
    
    subprocess.run(commande)
    actualiser_compteur()


# --- FONCTIONS ASSOCIÉES AUX BOUTONS ---
def action_capturer():
    """Bouton 1 : Lance panda2.py pour prendre une nouvelle mesure avec l'ESP32."""
    messagebox.showinfo(
        "Capture Lidar",
        "La capture va démarrer.\n\n"
        "Regardez votre terminal pour :\n"
        "1. Attendre la fin du tour Lidar\n"
        "2. Saisir le déplacement (dx dy dtheta)"
    )
    lancer_commande("panda2.py", [], "Capture d'un nouveau scan Lidar...")


def action_recaler():
    """Bouton 2 : Lance viewer.py exactement comme en ligne de commande."""
    fichiers = compter_photos()
    if not fichiers:
        messagebox.showwarning("Aucun scan", "Aucune photo trouvée dans le dossier photos/ !")
        return
    
    # On force le script à se lancer avec l'option --photos pour qu'il lise 
    # exactement les mêmes repères et déplacements que d'habitude.
    lancer_commande("viewer.py", ["--photos"], "Ouverture du visualiseur de recalage...")


def action_grid_map():
    """Bouton 3 : Lance grid_map.py (Carte d'occupation + Localisation Brute Force)."""
    fichiers = compter_photos()
    if not fichiers:
        messagebox.showwarning("Aucun scan", "Aucune photo trouvée dans le dossier photos/ !")
        return
    
    lancer_commande("grid_map_new.py", ["--photos"], "Calcul de la Grid Map et test de localisation...")


def action_vector_map():
    """Bouton 4 : Lance vector_map.py (Extraction des murs en segments)."""
    fichiers = compter_photos()
    if not fichiers:
        messagebox.showwarning("Aucun scan", "Aucune photo trouvée dans le dossier photos/ !")
        return
    
    lancer_commande("vector_map.py", ["--photos"], "Construction de la Carte Vectorielle...")


def action_slam_continu():
    """Bouton 5 : Lance le SLAM en continu."""
    # Avec le robot branché, on lance "slam_continu.py"
    lancer_commande("slam_continu.py", [], "Lancement du SLAM continu (simulation)...")


def action_simu_slam():
    """Bouton 6 : Lance le SLAM en simulation."""
    # Sans robot, on lance la simulation.
    lancer_commande("simu_slam.py", [], "Lancement du SLAM continu (simulation)...")


# =====================================================================
# CONSTRUCTION DE LA FENÊTRE GRAPHIQUE (TKINTER)
# =====================================================================
fenetre = tk.Tk()
fenetre.title("Centre de Contrôle Lidar")
fenetre.geometry("420x500")
fenetre.configure(bg="#2b2b2b")
fenetre.resizable(False, False)

# Titre principal
lbl_titre = tk.Label(
    fenetre, text="🤖 Tableau de Bord Lidar", 
    font=("Arial", 16, "bold"), bg="#2b2b2b", fg="#a6e22e", pady=15
)
lbl_titre.pack()

# Compteur de scans disponibles
lbl_statut = tk.Label(
    fenetre, text="Scans disponibles : 0", 
    font=("Consolas", 10), bg="#1e1e1e", fg="#66d9ef", padx=10, pady=6
)
lbl_statut.pack(fill="x", padx=30, pady=(0, 15))

# Style commun pour tous les boutons d'action
style_bouton = {
    "font": ("Arial", 11, "bold"),
    "width": 34,
    "pady": 8,
    "bd": 0,
    "cursor": "hand2"
}

# Création des 5 boutons principaux
btn_capture = tk.Button(
    fenetre, text="📸 1. Capturer un scan (Lidar)", 
    bg="#fd971f", fg="#1e1e1e", command=action_capturer, **style_bouton
)
btn_capture.pack(pady=6)

btn_viewer = tk.Button(
    fenetre, text="🎯 2. Recaler les scans (Viewer)", 
    bg="#66d9ef", fg="#1e1e1e", command=action_recaler, **style_bouton
)
btn_viewer.pack(pady=6)

btn_grid = tk.Button(
    fenetre, text="🧱 3. Générer la Grid Map & Localiser", 
    bg="#a6e22e", fg="#1e1e1e", command=action_grid_map, **style_bouton
)
btn_grid.pack(pady=6)

btn_vector = tk.Button(
    fenetre, text="📐 4. Générer la Carte Vectorielle", 
    bg="#ae81ff", fg="#1e1e1e", command=action_vector_map, **style_bouton
)
btn_vector.pack(pady=6)

btn_slam = tk.Button(
    fenetre, text=" 5. Lancer le slam continu", 
    bg="#ff00d9", fg="#1e1e1e", command=action_slam_continu, **style_bouton
)
btn_slam.pack(pady=6)

btn_simu = tk.Button(
    fenetre, text=" 6. Lancer la simulation", 
    bg="#ffff00", fg="#1e1e1e", command=action_simu_slam, **style_bouton
)
btn_simu.pack(pady=6)

# Bouton Quitter en bas
btn_quitter = tk.Button(
    fenetre, text="Quitter", font=("Arial", 10),
    bg="#f92672", fg="white", width=15, pady=4, bd=0, command=fenetre.destroy
)
btn_quitter.pack(pady=20)

# Initialisation du compteur au démarrage et lancement de la boucle d'affichage
actualiser_compteur()
fenetre.mainloop()