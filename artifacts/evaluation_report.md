# Neural CAT Test Evaluation Report

- **Evaluated Checkpoint**: `checkpoints/best-neural-cat-optimized-v14.ckpt`
- **Total Test Sequences**: `89192`
- **Total Interaction Steps Evaluated**: `1050110`

## Overall Metrics

| Metric | Value | Percentage |
| --- | --- | --- |
| **Accuracy** | `0.8041` | 80.41% |
| **AUC-ROC** | `0.7495` | - |
| **Precision (Lớp 1)** | `0.8576` | 85.76% |
| **Recall (Lớp 1)** | `0.9066` | 90.66% |
| **F1-Score (Lớp 1)** | `0.8814` | 88.14% |

## Detailed Classification Report

```text
                  precision    recall  f1-score   support

 Lớp 0 (Làm sai)       0.50      0.39      0.44    206771
Lớp 1 (Làm đúng)       0.86      0.91      0.88    843339

        accuracy                           0.80   1050110
       macro avg       0.68      0.65      0.66   1050110
    weighted avg       0.79      0.80      0.79   1050110

```

## Confusion Matrix

```text
[[ 79809 126962]
 [ 78792 764547]]
  - TN (Đoán đúng học sinh làm sai): 79809
  - FP (Đoán sai thành làm đúng):   126962
  - FN (Đoán sai thành làm sai):    78792
  - TP (Đoán đúng học sinh làm đúng): 764547
```

## Model Parameters Analysis

- **Average Guessing parameter ($g$)**: `0.0000`
- **Average Slip parameter ($s$)**: `0.0000`

> [!NOTE]
> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.
