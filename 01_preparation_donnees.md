# Préparation des données — tf_flowers (Daisy / Rose / Tulip)

Notebook concerné : `prepare_data_v2.ipynb`
Fichier produit : `tf_flowers_prepared.npz`

---

## 1. Objectif

Transformer le dataset brut `tf_flowers` (5 classes, images de tailles variables) en un jeu de données propre, au format fixe, découpé en **train / validation / test**, utilisable par les trois modèles (Softmax, k-NN, CNN).

On ne garde que 3 classes : **daisy**, **roses**, **tulips**.

---

## 2. Pipeline en un coup d'œil

```
tf_flowers (3 670 images, 5 classes)
        │
        ▼
Exploration (distribution des classes, contrôle visuel)
        │
        ▼
Filtrage : daisy, roses, tulips  (≈ 2 073 images)
        │
        ▼
Resize 128×128 (antialias) → uint8
        │
        ▼
Split stratifié 70 / 15 / 15
        │
        ▼
Contrôle des doublons entre splits
        │
        ▼
Sauvegarde  tf_flowers_prepared.npz
```

---

## 3. Détail des étapes

### 3.1 Chargement
Le dataset est chargé avec `tensorflow_datasets` (`tfds.load('tf_flowers', as_supervised=True)`). Chaque élément est une paire `(image, label)`.

### 3.2 Exploration
- **Distribution des classes** : histogramme du nombre d'images par classe sur le dataset complet.
- **Contrôle visuel** : grille de 6 images aléatoires par classe, pour vérifier que les labels correspondent bien aux images après filtrage.

### 3.3 Filtrage des classes
Les labels d'origine (ex. `daisy = 1`, `tulips = 2`, `roses = 4`) sont remappés vers `0, 1, 2` :

| Label final | Classe | Nom d'affichage |
|---|---|---|
| 0 | daisy | Daisy |
| 1 | roses | Rose |
| 2 | tulips | Tulip |

Une **seule liste de noms** (`CLASS_NAMES`) est utilisée partout, pour éviter les incohérences d'affichage entre notebooks.

### 3.4 Redimensionnement
- Taille cible : **128 × 128** pixels (`IMG_SIZE`).
- `tf.image.resize(..., antialias=True)` : évite les artefacts quand on réduit fortement des images de ~500×333.
- Arrondi puis conversion en **`uint8`** (valeurs 0–255).

> Remarque : le resize déforme légèrement les images non carrées. C'est acceptable ici ; une alternative est de recadrer au centre avant le resize, au risque de couper une partie de la fleur.

### 3.5 Split stratifié 70 / 15 / 15
Deux `train_test_split` successifs avec `stratify` et `random_state=42` :

| Split | Proportion | Rôle |
|---|---|---|
| Train | 70 % (~1 451 images) | Apprentissage des paramètres |
| Validation | 15 % (~311 images) | Choix des hyper-paramètres, early stopping, poids du vote |
| Test | 15 % (~311 images) | Évaluation finale, **utilisé une seule fois** |

La stratification garantit les mêmes proportions de classes dans les trois splits.

### 3.6 Contrôle anti-fuite de données
On calcule un hash de chaque image et on vérifie que l'intersection train∩val, train∩test et val∩test est **vide**. Une image présente à la fois dans train et test fausserait l'accuracy.

### 3.7 Sauvegarde
`np.savez_compressed` écrit :

| Clé | Contenu |
|---|---|
| `X_train`, `X_val`, `X_test` | images `uint8`, forme `(N, 128, 128, 3)` |
| `y_train`, `y_val`, `y_test` | labels entiers 0, 1, 2 |
| `class_names` | `['Daisy', 'Rose', 'Tulip']` |

---

## 4. Ce qui a été corrigé par rapport à la version précédente

| Problème | Conséquence | Correction |
|---|---|---|
| 600 images/classe seulement | ~1 800 images au lieu de ~2 070, donc moins de données et plus d'overfitting | Toutes les images des 3 classes sont conservées |
| Split 60/20/20 | Seulement ~1 080 images d'entraînement | Split 70/15/15 |
| Resize sans anti-aliasing | Perte de qualité, images crénelées | `antialias=True` |
| Stockage en `float32` | Mémoire et fichier ×4 | Stockage en `uint8`, normalisation `/255` faite à l'entraînement |
| Commentaires faux (« 32x32 », « 300 images ») | Confusion | Commentaires et variables alignés avec le code |
| Noms de classes incohérents (`daisy/roses/tulips` vs `Daisy/Rose/Tulip`) | Risque d'erreur d'affichage | Liste unique `CLASS_NAMES` |
| Déséquilibre ignoré (tulips ≈ 25 % d'images en plus) | Biais vers la classe majoritaire | `class_weight` calculés dans le notebook de modélisation |

---

## 5. Points d'attention

- **Normalisation** : faite dans le notebook d'entraînement (`/255`). Les statistiques éventuelles (moyenne, écart-type, PCA) sont calculées **uniquement sur le train** pour éviter toute fuite.
- **Augmentation de données** : appliquée **uniquement** au train, à l'intérieur du CNN (couches Keras actives seulement pendant `fit`). Ni la validation ni le test ne sont augmentés.
- **Reproductibilité** : `SEED = 42` pour NumPy, TensorFlow et les splits.
- **Quasi-doublons** : tf_flowers contient quelques images très proches (non identiques). Le contrôle par hash ne les détecte pas. Impact limité, mais à garder en tête si le score test paraît trop beau.

---

## 6. Pistes d'amélioration possibles

1. Recadrage centré avant resize (éviter la déformation).
2. Utiliser aussi `dandelion` et `sunflowers` comme données d'entraînement supplémentaires (si le sujet l'autorise).
3. Validation croisée stratifiée en 5 plis pour des résultats moins dépendants d'un seul split.
4. Détection de quasi-doublons (hash perceptuel) avant le split.
