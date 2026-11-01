# Data Preparation Pipeline – TF Flowers

## 1. Objective

The objective of data preparation is to transform the original TF Flowers dataset into a clean, balanced, and standardized dataset ready for training and evaluating three classification models:

- k-Nearest Neighbors (k-NN)
- Linear Softmax Classifier
- Convolutional Neural Network (CNN)

We select only three flower categories: **Daisy, Rose, and Tulip**.

# Understand the data preparation pipeline

             TF Flowers Dataset
                     |
                     v
             Load all images
                     |
                     v
          Select 3 required classes
          Daisy / Rose / Tulip
                     |
                     v
          Select 300 images per class
                     |
                     v
              900 images
                     |
                     v
           Resize images to 32x32
                     |
                     v
             RGB normalization
                     |
                     v
          Stratified Data Splitting
                     |
          +----------+----------+
          |          |          |
          v          v          v
        Train      Validation   Test
         60%         20%        20%
          |          |          |
          +----------+----------+
                     |
                     v
              Ready for ML models




## 2. Loading and Selecting Data

We load the TF Flowers dataset using TensorFlow Datasets (TFDS).

The original dataset contains five flower categories. We keep only the three required classes.

We assign a numerical label to each class:

| Class | Label |
|---|---:|
| Daisy | 0 |
| Rose | 1 |
| Tulip | 2 |

**Why?** Machine learning models work with numerical labels rather than text categories.

## 3. Image Resizing

All selected images are resized to **32 × 32 × 3**:

- 32: image height.
- 32: image width.
- 3: RGB color channels.

**Why?** Images must have identical dimensions to be processed by the models. A small resolution also reduces memory usage and computation time.

## 4. Dataset Balancing

We randomly select exactly **300 images per class**, giving us 900 images in total.

| Class | Images |
|---|---:|
| Daisy | 300 |
| Rose | 300 |
| Tulip | 300 |
| **Total** | **900** |

We use a fixed random seed (42) to make the selection reproducible.

**Why?** A balanced dataset prevents the models from favoring a class simply because it contains more images.

## 5. Normalization

Original RGB pixel values range from 0 to 255.

We divide every pixel value by 255 to obtain values between 0 and 1.

\[
X_{\text{normalized}} = \frac{X}{255}
\]

**Why?** Normalization puts input values on a consistent scale and helps neural networks train more stably.

## 6. Dataset Splitting

We divide the 900 images into three subsets using stratified splitting.

| Dataset | Percentage | Images | Images per class |
|---|---:|---:|---:|
| Training | 60% | 540 | 180 |
| Validation | 20% | 180 | 60 |
| Test | 20% | 180 | 60 |
| **Total** | **100%** | **900** | **300** |

**Purpose of each subset:**

- **Training:** Used to train the models.
- **Validation:** Used to select hyperparameters and monitor performance.
- **Test:** Used only for the final evaluation on unseen images.

**Why stratified splitting?** It preserves the same class proportions in all three subsets.

**Important:** We must not use the test set to select hyperparameters because this could make the final evaluation unreliable.

## 7. Data Verification and Visualization

Before training, we:

- Check the number of images in each subset.
- Verify image dimensions and pixel ranges.
- Check the distribution of classes.
- Display sample images from each category.

**Why?** These checks help detect incorrect labels, corrupted images, unexpected dimensions, or class imbalance.

## 8. Saving the Prepared Dataset

We save the prepared data in `tf_flowers_prepared.npz`.

The file contains:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- Class names

**Why?** Saving the dataset avoids repeating preprocessing and ensures that all three models use exactly the same data and splits.

## 9. Final Dataset Summary

At the end of preparation, we obtain:

| Property | Expected Result |
|---|---|
| Total images | 900 |
| Classes | 3 |
| Images per class | 300 |
| Image shape | 32 × 32 × 3 |
| Training set | 540 |
| Validation set | 180 |
| Test set | 180 |
| Pixel range | [0, 1] |

## 10. Conclusion

The data preparation pipeline produces a balanced and standardized dataset ready for classification.

All three models use the same prepared images, ensuring a fair comparison between k-NN, Linear Softmax, and CNN.