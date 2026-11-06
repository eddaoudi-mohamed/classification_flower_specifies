# Modélisation — Softmax, k-NN, CNN et modèle de vote

Notebook concerné : `flowers_all_models.ipynb`
Entrée : `tf_flowers_prepared.npz` (voir `01_preparation_donnees.md`)
Tâche : classification de 3 classes de fleurs (Daisy, Rose, Tulip)

---

## 1. Vue d'ensemble

| # | Modèle | Entrée | Idée |
|---|---|---|---|
| 1 | Softmax (régression logistique multinomiale) | image 32×32 | Classifieur linéaire, base de comparaison |
| 2 | k-NN | image 32×32 + PCA | Classe = celle des voisins les plus proches |
| 3 | CNN | image 128×128 | Apprend lui-même des filtres visuels |
| 4 | Vote (dur, doux, pondéré) | sorties des modèles 1 à 3 | Combine les trois modèles |

**Protocole d'évaluation**
- **Train** : apprentissage.
- **Validation** : choix des hyper-paramètres, early stopping, poids du vote.
- **Test** : évaluation finale uniquement.

Métriques calculées pour chaque modèle : accuracy (train, validation, test), précision, rappel et F1 (moyenne macro sur le test), F1 par classe, matrice de confusion. L'**écart train − test** sert d'indicateur d'overfitting.

---

## 2. Pourquoi l'ancienne version overfittait

| Modèle | Cause | Effet |
|---|---|---|
| CNN | `Flatten` sur une carte 36×36×64 = 82 944 valeurs, puis `Dense(64)` : environ **5,3 M de paramètres** pour ~1 000 images | Le modèle mémorise le train |
| « Softmax » | C'était un MLP (32-16) sur 67 500 pixels bruts, pas un softmax simple | Beaucoup de paramètres, peu de données |
| k-NN | Distances sur 67 500 dimensions | Malédiction de la dimension : les distances deviennent peu discriminantes |
| Données | Seulement ~1 080 images d'entraînement | Trop peu pour généraliser |

---

## 3. Préparation commune (chargement)

- Pixels convertis en `float32` dans [0, 1].
- Dimensions lues depuis les données (`INPUT_SHAPE`), donc le code fonctionne aussi avec des images 150×150.
- **`class_weight`** (`'balanced'`) calculés sur le train : la classe Tulip, plus nombreuse, ne domine pas l'apprentissage.

---

## 4. Modèle 1 — Softmax

### Architecture
```
Input (128,128,3)
  → Normalization (moyenne/écart-type par canal, calculés sur le train)
  → AveragePooling2D(4)        # 128 → 32
  → Flatten                    # 3 072 valeurs
  → Dense(3, softmax, L2=1e-3)
```

### Choix
- **Une seule couche** : c'est une vraie régression softmax. Environ 9 000 paramètres contre 2,2 M pour l'ancien MLP.
- **Réduction à 32×32** : moins d'entrées, donc moins de risque de mémoriser.
- **L2 = 1e-3** : pénalise les poids trop grands.

### Entraînement
- Adam (lr = 1e-3), batch 32, jusqu'à 100 époques.
- `EarlyStopping` (surveille `val_loss`, patience 12, restaure les meilleurs poids).
- `ReduceLROnPlateau` (×0,5 après 4 époques sans progrès).
- `class_weight` activé.

### Rôle attendu
Modèle de référence : s'il fait à peine mieux que le hasard (33 %), les pixels bruts sont peu informatifs, ce qui est normal pour un classifieur linéaire sur des photos.

---

## 5. Modèle 2 — k-NN

### Pipeline
```
Image 128×128 → réduction 32×32 (antialias) → vecteur 3 072
              → PCA (n composantes, ajustée sur le train)
              → KNeighborsClassifier
```

### Recherche d'hyper-paramètres (sur la validation)

| Paramètre | Valeurs testées |
|---|---|
| PCA | pixels bruts, 50, 100 composantes |
| K | 1, 3, 5, 7, 9, 11, 15, 21 |
| Distance | euclidienne (L2), Manhattan (L1), cosinus |
| Pondération | uniforme, par distance |

Soit 144 combinaisons. Le notebook affiche le top 10 et la meilleure configuration.

### Graphiques produits
- Accuracy de validation en fonction de K (une courbe par distance/pondération).
- Effet de la PCA (meilleure accuracy de validation pour chaque option).

### Points à connaître
- **Pas d'accuracy train pour k-NN** : chaque point d'entraînement serait son propre voisin, la valeur serait artificiellement haute et trompeuse. Elle est donc laissée vide dans les tableaux.
- k-NN ne « s'entraîne » pas : `fit` mémorise simplement les données.
- Le k-NN sur pixels reste en général le modèle le plus faible sur des photos naturelles.

---

## 6. Modèle 3 — CNN

### Architecture
```
Input (128,128,3)
  → Augmentation (flip horizontal, rotation 0,10, zoom 0,15, contraste 0,15)
  → [Conv3×3 → BatchNorm → ReLU → MaxPool2] × 4   # filtres : 32, 64, 128, 256
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(64, ReLU, L2=1e-4)
  → Dropout(0.3)
  → Dense(3, softmax)
```
Environ **400 000 paramètres** (contre 5,3 M avant).

### Pourquoi ces choix

| Élément | Rôle |
|---|---|
| `GlobalAveragePooling` à la place de `Flatten` | Supprime la plus grosse source de paramètres, donc d'overfitting |
| `BatchNorm` | Entraînement plus stable et plus rapide, léger effet régularisant |
| Augmentation | Le modèle voit des variantes de chaque image à chaque époque ; active uniquement pendant l'entraînement |
| Dropout (0,4 puis 0,3) + L2 | Régularisation supplémentaire |
| 4 blocs au lieu de 2 | Plus de capacité de représentation avec peu de paramètres |

### Entraînement
- Adam (lr = 1e-3), batch 32, jusqu'à 80 époques.
- `EarlyStopping` (`val_loss`, patience 12, `restore_best_weights=True`).
- `ReduceLROnPlateau` (×0,5, patience 4, minimum 1e-5).
- `class_weight` activé.
- Les accuracy train/val/test sont recalculées après entraînement, **sans augmentation**.

---

## 7. Modèle de vote (ensemble)

On combine les probabilités ou les prédictions de Softmax, k-NN et CNN.

| Variante | Principe |
|---|---|
| **Vote dur** | Chaque modèle vote pour une classe, la majorité gagne. Si les trois sont en désaccord, on suit le modèle ayant la meilleure accuracy de validation. |
| **Vote doux** | Moyenne simple des trois vecteurs de probabilités, puis `argmax`. |
| **Vote pondéré** | Moyenne pondérée. Les poids (somme = 1, pas de 0,1) sont choisis par recherche exhaustive sur la **validation** : meilleure accuracy, départage par la log-loss. |

### À savoir
- Les poids sont choisis **sans regarder le test**, donc l'évaluation reste honnête.
- Un vote n'améliore pas toujours le meilleur modèle seul : si le CNN est nettement meilleur que les deux autres, les poids optimaux lui donneront presque tout (un poids de 0 pour un modèle est possible).
- Les probabilités de k-NN sont grossières (fractions de voisins), ce qui limite l'intérêt du vote doux.

---

## 8. Comment lire les résultats

Le notebook produit un tableau et quatre familles de graphiques.

| Sortie | À regarder |
|---|---|
| Courbes accuracy/loss (Softmax, CNN) | Si la courbe de validation s'éloigne de celle du train, il y a overfitting |
| Tableau récapitulatif | Colonne **Écart Train-Test** : plus elle est faible, mieux le modèle généralise |
| Barres Train/Val/Test | Visualisation directe de l'overfitting |
| Barres Précision/Rappel/F1 | Vérifier que l'accuracy n'est pas portée par une seule classe |
| Matrices de confusion normalisées | Quelles classes sont confondues (souvent Rose ↔ Tulip) |
| Heatmap F1 par classe | Classe la plus difficile pour chaque modèle |
| 10 erreurs du meilleur modèle | Comprendre les échecs (fleur mal cadrée, fond chargé, etc.) |

**Repères**
- Écart train-test inférieur à environ 5 points : généralisation correcte.
- Écart supérieur à environ 15 points : overfitting marqué, à corriger (plus de régularisation ou d'augmentation, modèle plus petit, plus de données).

Les fichiers `results_summary.csv` et `cnn_flowers.keras` sont écrits à la fin du notebook.

---

## 9. Pistes pour améliorer l'accuracy

1. **Transfert d'apprentissage** (MobileNetV2 ou EfficientNetB0 pré-entraîné sur ImageNet, base gelée, petite tête de classification) : c'est le gain le plus important possible avec ~2 000 images. À utiliser seulement si le sujet l'autorise.
2. **Features de couleur** pour k-NN et Softmax : histogrammes HSV concaténés aux pixels réduits, car la couleur discrimine bien daisy, rose et tulip.
3. **Plus de données** : classes supplémentaires en entraînement, augmentation plus forte (`RandomTranslation`, `RandomBrightness`).
4. **Validation croisée stratifiée** (5 plis) : résultats plus fiables qu'un seul split de ~311 images.
5. **Test-time augmentation** : moyenner les prédictions du CNN sur l'image et sa version retournée.
6. **Réglage fin du CNN** : taux d'apprentissage, valeurs de dropout, nombre de filtres, `label_smoothing` (nécessite la perte `CategoricalCrossentropy` avec labels one-hot).

---

## 10. Hyper-paramètres principaux (récapitulatif)

| Paramètre | Valeur |
|---|---|
| Graine aléatoire | 42 |
| Taille d'image (CNN) | 128×128 |
| Résolution Softmax / k-NN | 32×32 |
| Batch size | 32 |
| Époques max Softmax / CNN | 100 / 80 |
| Optimiseur | Adam, lr = 1e-3 |
| EarlyStopping | `val_loss`, patience 12 |
| ReduceLROnPlateau | facteur 0,5, patience 4, min 1e-5 |
| Dropout CNN | 0,4 puis 0,3 |
| Régularisation L2 | 1e-3 (Softmax), 1e-4 (CNN) |
