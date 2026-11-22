# Compte rendu — Cartographie et Localisation 2D (SLAM Lidar)

## 1. Dépôt Git et historique des commits

Ce projet a été développé et versionné sur le dépôt GitHub.

L'historique des commits retrace l'évolution progressive du travail : acquisition des scans Lidar, recalage des poses, construction de la grid map, extraction de la carte vectorielle, mise en place de la localisation, interface graphique, puis passage au mode continu et à la simulation. 

Les images et versions intermédiaires de fichiers sont directement consultables dans l'historique des commits du dépôt.

Les commentaires concernant les choix apportés et la justification des modifications se trouvent dans la description de chaque commit.

## 2. Outils développés et instructions de test

### Prérequis

Installer les bibliothèques nécessaires dans l'environnement Python :

```
pip install numpy matplotlib pandas pyserial

```

### Présentation et lancement de chaque outil

1. **`scan_io.py` — Lecture des scans et changements de repère**

   * Définit la structure `Scan`, le changement de repère entre le Lidar et le monde réel, le chargement des CSV (`load_scan`, `load_photo_csv`) et l'extraction du premier tour de Lidar (`premier_tour`).

2. **`panda2.py` — Capture série et gestion des déplacements**

   * Lit les trames envoyées par l'ESP32 (`PORT_ARDUINO`, `BAUDRATE`), vérifie que les scans sont complets (`scan_valide`), enregistre les captures dans `photos/photo_XXX.csv` et cumule les déplacements (`calculer_poses`).

   * **Test** :

     ```
     python panda2.py
     python panda2.py --show
     
     ```

3. **`viewer.py` — Visualisation et recalage interactif**

   * Affiche les nuages de points des différents scans et permet de modifier $(x, y, \theta)$ au clavier pour les recaler et enregistrer les poses.

   * **Test** :

     ```
     python viewer.py --photos
     python viewer.py scans/*.csv --poses poses.json
     
     ```

4. **`grid_map_new.py` — Grid Map et localisation brute force**

   * Met à jour une grid map discrète par lancer de rayons (`bresenham`, `mettre_a_jour_carte`) et estime la pose du robot autour d'une pose de départ (`evaluer_score`, `localisation_brute_force`).

   * **Test** :

     ```
     python grid_map_new.py --photos
     python grid_map_new.py scans/*.csv --poses poses.json
     
     ```

5. **`vector_map.py` — Extraction et fusion de segments (Carte vectorielle)**

   * Regroupe les points par clusters adaptatifs, découpe par Douglas-Peucker, ajuste des droites par moindres carrés (`ajuster_segment`), filtre les segments courts/bruités et fusionne les segments proches (`inserer`, `fusionner`).

   * **Test** :

     ```
     python vector_map.py --photos
     python vector_map.py --photos --min-obs 2
     
     ```

6. **`interface.py` — Interface graphique de contrôle**

   * Fenêtre Tkinter regroupant le suivi des scans disponibles et des boutons pour lancer les différents scripts du projet.

   * **Test** :

     ```
     python interface.py
     
     ```

7. **`slam_continu.py` et `simu_slam.py` — Cartographie et localisation en continu**

   * `slam_continu.py` lit le port série en continu dans un thread avec une `queue.Queue`, estime/recale la pose à chaque tour complet et met à jour en direct la `GridMap` et la `CarteVectorielle`.

   * `simu_slam.py` simule les scans Lidar dans une pièce en L avec des obstacles pour tester la boucle continue sans matériel.

   * **Test** :

     ```
     python simu_slam.py
     python slam_continu.py --port COM5
     
     ```

## 3. Conclusion personnelle sur le module

### Synthèse par outil

* **`scan_io.py`**

  * *Fonctionnalités et méthodes* : classe `Scan`, `points_local()`, `points_world()`, `load_scan()`, `load_photo_csv()` et `premier_tour()`.

  * *Choix effectués* : les angles sont passés en sens antihoraire, les distances hors de la plage définie sont filtrées. Cela est fait dès la chargement pour avoir des données uniformes et propres sur lesquelles travailler. 

  * *Résultats visuels* : des nuages de points  sans mesures nulles de répétitions sur plusieurs tours.

* **`panda2.py`**

  * *Fonctionnalités et méthodes* : `lire_ligne()`, `scan_valide()`, `capturer_photo()`, `lire_deplacements()`, `calculer_poses()` et `tracer_carte()`.

  * *Choix effectués* : on rejette les trames qui sont tronquées ou en double et on stock les déplacements dans un fichier CSV pour pouvoir retracer la carte avec `--show`.

  * *Résultats visuels* : les scan et les trajectoires  enregistrés sont superposés graphiquement dans `photos/carte.png`.

* **`viewer.py`**

  * *Fonctionnalités et méthodes* : classe `Viewer` avec gestion des événements clavier (`on_key`), mise à jour de l'affichage (`update`), cadrage (`fit`) et sauvegarde (`save_poses`, `save_deplacements`).

  * *Choix effectués* : le premier scan est fixé comme origine en mode `--photos` pour garder un repère stable. Des options sont ajoutées pour la modification de la carte (déplacements des scan, vitesse, …).

  * *Résultats visuels* : alignement visuel des nuages de points de chaque scan avec affichage des poses et orientations.

* **`grid_map_new.py`**

  * *Fonctionnalités et méthodes* : `bresenham()`, `GridMap` (`world_to_grid`, `in_bounds`), `mettre_a_jour_carte()`, `evaluer_score()` et `localisation_brute_force()`.

  * *Choix effectués* : grid map avec 3 états/couleurs ($0$ libre, $127$ inconnu, $255$ mur), une résolution fixe ($0.05\text{ m}$) et une recherche locale autour de la pose estimée pour rester simple et lisible.

  * *Résultats visuels* : obtention d'une carte probabiliste et d'une carte seuillée qui montrent l'espace libre et les murs, ainsi qu'un recalage de pose.

* **`vector_map.py`**

  * *Fonctionnalités et méthodes* : `Segment` (`distance_droite`, `angle_avec`, `est_proche`, `fusionner`, `intersection`, `decouper`), `douglas_peucker_indices()`, `ajuster_segment()` et `CarteVectorielle` (`clusters`, `extraire_segments`, `inserer`, `mettre_a_jour`, `nettoyer`, `sauver`).

  * *Choix effectués* : adapter l'écart entre points voisins à la distance, filtrer les segments trop courts ou trop bruités (`min_len`, `rms_max`) et fusionner les segments proches par moyenne pondérée.

  * *Résultats visuels* : extraction de segments représentant les murs et obstacles, affichés en orange (vus une fois) ou en bleu (confirmés sur plusieurs observations) dans `carte_vectorielle.png`.

* **`interface.py`**

  * *Fonctionnalités et méthodes* : `compter_photos()`, `actualiser_compteur()`, `lancer_commande()` et fonctions associées aux boutons de la fenêtre Tkinter.

  * *Choix effectués* : appeler chaque script via `subprocess.run` afin de garder un comportement identique à l'exécution en ligne de commande.

  * *Résultats visuels* : tableau de bord centralisé permettant d'accéder facilement à toutes les étapes du projet.

* **`slam_continu.py` et `simu_slam.py`**

  * *Fonctionnalités et méthodes* : lecture série dans un thread (`worker_serie`), file d'attente `queue.Queue`, mise à jour continue de `GridMap` et `CarteVectorielle` avec `localisation_brute_force`, et simulation de scans (`construire_piece_en_L`, `distance_rayon_segment`, `simuler_scan_piece`).

  * *Choix effectués* : utilise la même base de classes que dans les scripts précédents. Pour **`simu_slam.py`** j’ajoute une simulation d'environnement en L avec obstacles pour valider le fonctionnement continu sans matériel.

  * *Résultats visuels* : mise à jour progressive de la grille d'occupation, de la carte vectorielle et de la trajectoire du robot (`slam_continu.png` et captures de simulation).

### Récapitulatif du module

* **Ce que j'ai apprécié** : la progression étape par étape du TP. Malgré une forte utilisation de l’IA, cette manière de progresser a permis de comprendre les concepts et d’apprendre à utiliser des outils. Outre l’apprentissage du SLAM, j’ai apprécié l’utilisation de GitHub même si je m’y suis mis tard, c’est un outils qui me servira surement à l’avenir. Concernant le SLAM en lui-même, comprendre la démarche derrière, les différents “outils” qui la compose et arriver au final à un script permettant de se localiser et de cartographier en temps réel (même si je n’ai pas eu l’occasion de tester mon dernier script) fu très intéressant. 

* **Ce que j'ai moins apprécié** : le manque de robot a rendu l’expérimentation compliquée. Je pense que réaliser le travail réellement en groupe et donc avoir un seul rendu par groupe aurait rendu la chose plus agréable. 

* **Ce que j'améliorerais si j'en avais la possibilité** : concernant mon travail, ce serait surement effectuer plus de tests et retravailler ma carte vectorielle que je ne trouve pas satisfaisante. Concernant le module, selon moi seul le manque de robot pose problème. 
