# Neural CAT Test Evaluation Report

- **Evaluated Checkpoint**: `checkpoints/best-neural-cat-epoch=20-val_loss=0.462.ckpt`
- **Total Test Sequences**: `89192`
- **Total Interaction Steps Evaluated**: `1050129`

## Overall Metrics

| Metric | Value | Percentage |
| --- | --- | --- |
| **Accuracy** | `0.8094` | 80.94% |
| **AUC-ROC** | `0.7061` | - |
| **Precision** | `0.8121` | 81.21% |
| **Recall** | `0.9923` | 99.23% |
| **F1-Score** | `0.8932` | 89.32% |

## Model Parameters Analysis

- **Average Guessing parameter ($g$)**: `0.0588`
- **Average Slip parameter ($s$)**: `0.1908`

> [!NOTE]
> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.
