# Neural CAT Test Evaluation Report

- **Evaluated Checkpoint**: `checkpoints/best-neural-cat-optimized-epoch=48-val_loss=0.256.ckpt`
- **Total Test Sequences**: `68`
- **Total Interaction Steps Evaluated**: `762`

## Overall Metrics

| Metric | Value | Percentage |
| --- | --- | --- |
| **Accuracy** | `0.9226` | 92.26% |
| **AUC-ROC** | `0.8717` | - |
| **Precision** | `0.9483` | 94.83% |
| **Recall** | `0.9663` | 96.63% |
| **F1-Score** | `0.9572` | 95.72% |

## Model Parameters Analysis

- **Average Guessing parameter ($g$)**: `0.0649`
- **Average Slip parameter ($s$)**: `0.1458`

> [!NOTE]
> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.
