# Neural CAT Test Evaluation Report

- **Evaluated Checkpoint**: `checkpoints/best-neural-cat-optimized-v6.ckpt`
- **Total Test Sequences**: `89192`
- **Total Interaction Steps Evaluated**: `1050129`

## Overall Metrics

| Metric | Value | Percentage |
| --- | --- | --- |
| **Accuracy** | `0.8105` | 81.05% |
| **AUC-ROC** | `0.7044` | - |
| **Precision** | `0.8187` | 81.87% |
| **Recall** | `0.9814` | 98.14% |
| **F1-Score** | `0.8927` | 89.27% |

## Model Parameters Analysis

- **Average Guessing parameter ($g$)**: `0.0573`
- **Average Slip parameter ($s$)**: `0.0159`

> [!NOTE]
> The Guessing and Slip parameters show the model's calibration. An average guessing parameter under 0.2 and slip parameter under 0.15 indicates strong item response calibration on the test data.
