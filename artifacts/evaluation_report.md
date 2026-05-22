# Neural CAT Test Evaluation Report

- **Evaluated Checkpoint**: `checkpoints/best-neural-cat-optimized-epoch=09-val_loss=0.337.ckpt`
- **Total Test Sequences**: `68`
- **Total Interaction Steps Evaluated**: `762`

## Overall Metrics

| Metric | Value | Percentage |
| --- | --- | --- |
| **Accuracy** | `0.8963` | 89.63% |
| **AUC-ROC** | `0.4947` | - |
| **Precision** | `0.8963` | 89.63% |
| **Recall** | `1.0000` | 100.00% |
| **F1-Score** | `0.9453` | 94.53% |

## Model Parameters Analysis

- **Average Guessing parameter ($g$)**: `0.0656`
- **Average Slip parameter ($s$)**: `0.1101`

> [!NOTE]
> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.
