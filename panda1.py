import matplotlib.pyplot as plt
import pandas as pd

# 1. Charger le CSV sans en-tête et nommer les colonnes à la volée
# sep=None permet de détecter automatiquement si c'est séparé par une virgule, un espace ou un point-virgule
df = pd.read_csv(
    "CSV.csv", header=None, names=["X", "Y"], sep=None, engine="python"
)

# 2. Créer le graphique LiDAR
# 's=1' réduit la taille des points pour que le nuage de points LiDAR soit bien lisible si vous avez beaucoup de données
plt.scatter(df["X"], df["Y"], color="black", s=1, marker=".")

# 3. Ajuster l'affichage pour du mapping (Mécatronique)
plt.title("Cartographie Nuage de Points - LiDAR")
plt.xlabel("Distance X")
plt.ylabel("Distance Y")
plt.grid(True)

# Force les axes X et Y à avoir la même échelle (très important pour que votre carte LiDAR ne soit pas déformée !)
plt.axis("equal")

# 4. Ouvrir la fenêtre interactive
plt.show()


