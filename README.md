# Pneumonia Classification from Chest X-rays: EfficientNetB0 + Grad-CAM

Three-class chest X-ray classifier (**Bacterial pneumonia / Normal / Viral pneumonia**) built with transfer learning on EfficientNetB0, with Grad-CAM heatmaps to show which image regions drive each prediction.

The whole pipeline lives in one notebook: `EfficientNet_GradCAM_Pneumonia_final.ipynb` (written for Google Colab, T4 GPU).

## Results

Held-out test set: 1,580 images.

| Stage | Setup | Test accuracy |
|---|---|---|
| 1 | Frozen EfficientNetB0 backbone, new classification head | 74.2% |
| 2 | Last 50 backbone layers unfrozen, low learning rate | **82.3%** |

Stage 2 classification report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial | 0.92 | 0.73 | 0.82 | 744 |
| Normal | 0.93 | 0.97 | 0.95 | 445 |
| Viral | 0.63 | 0.84 | 0.72 | 391 |
| **Macro avg** | 0.83 | 0.85 | 0.83 | 1,580 |

Stage 2 confusion matrix (rows = true, columns = predicted):

|  | Bacterial | Normal | Viral |
|---|---|---|---|
| **Bacterial** | 546 | 14 | 184 |
| **Normal** | 2 | 432 | 11 |
| **Viral** | 47 | 17 | 327 |

Normal cases are identified very reliably (97% recall). Most of the remaining error is bacterial vs. viral confusion: 184 bacterial cases are predicted as viral.

## Pipeline

1. **Data loading.** Two sources are merged:
   - A `train/val/test` chest X-ray dataset with `NORMAL` and `PNEUMONIA` folders. The `PNEUMONIA` images are split into `BACTERIAL` and `VIRAL` using the filename (`bacteria` / `virus`).
   - A Kaggle dataset with `labels_train.csv` and a `train_images/` folder. Its labels are remapped to match the label scheme below.
2. **Unified labels.** `0 = BACTERIAL`, `1 = NORMAL`, `2 = VIRAL`.
3. **Re-split.** All images (10,528 total) are pooled, shuffled with `random_state=42`, and split **70 / 15 / 15** into train / val / test.
4. **Preprocessing.** Resize to 224×224, RGB, EfficientNet `preprocess_input`, batch size 32.
5. **Class imbalance.** Inverse-frequency class weights, computed from the training set (train counts: bacterial 3,478, normal 1,965, viral 1,926).
6. **Model.** Augmentation (horizontal flip, rotation 0.1, zoom 0.1, contrast 0.2) → EfficientNetB0 (ImageNet weights, no top) → Global Average Pooling → Dropout 0.5 → Dense(3, softmax).
7. **Stage 1: frozen backbone.** Adam, lr `1e-3`, 50 epochs.
8. **Stage 2: fine-tuning.** Unfreeze the last 50 backbone layers, Adam lr `3e-5`, early stopping on `val_loss` (patience 5, restore best weights).
9. **Evaluation.** Accuracy, classification report, confusion matrix, one-vs-rest ROC curves.
10. **Explainability.** Grad-CAM on the `top_conv` layer, shown as heatmap + overlay next to the true and predicted labels.

## Outputs

Running the notebook saves:

| File | Description |
|---|---|
| `pneumonia_model.keras` | Trained model (saved to Google Drive) |
| `training_curves_stage1.png` | Loss/accuracy, frozen backbone |
| `training_curves_stage2.png` | Loss/accuracy, fine-tuning (best epoch marked) |
| `pipeline_diagram.png` | Pipeline overview figure |
| `class_dist_bar.png` | Class distribution |
| `confidence_distribution.png` | Prediction confidence distribution |

## Setup and usage

**Requirements:** Python 3, TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn, OpenCV (`opencv-python`). A GPU is strongly recommended.

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python
```

**Running on Colab (as written):**

1. Upload the two dataset archives to your Google Drive as `chest_xray.zip` and `chest_xray_2.zip`.
2. Open the notebook in Colab and set the runtime to **GPU**.
3. Run all cells. The notebook mounts Drive, copies and unzips the archives, builds the dataframes, trains both stages, evaluates, and generates Grad-CAM visualisations.

**Running locally:** replace the Colab-specific cells (Drive mounting, `!cp`, `!unzip`, `/content/...` paths) with paths to your local copies of the data.

## Limitations

- **Split method.** The merged data is shuffled and split at the *image* level. If the same patient has several images, they can land in both train and test, which can inflate the reported numbers. A patient-level split would give a more honest estimate. The two sources were also not checked for duplicate images.
- **Test set reuse.** The test set was evaluated after Stage 1 and again after Stage 2, and the final model should be treated as validated on this single split only. No cross-validation or external test set is included.
- **Bacterial vs. viral.** This is the hardest distinction (viral precision is 0.63). Treat the class-level scores, not just overall accuracy, as the real measure of performance.
- **Grad-CAM** highlights regions that influenced the prediction. It is not proof that the model is looking at clinically meaningful features.

## Disclaimer

This is a research and educational project. It is **not** a medical device and must not be used for clinical diagnosis or treatment decisions.

## Acknowledgements

- EfficientNet: Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (2019)
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)
- Datasets: add links and citations for the two chest X-ray datasets you used.