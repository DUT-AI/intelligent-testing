import os
import glob
import json
import pandas as pd

def main():
    # Find all JSON metrics files in checkpoints/
    metric_files = glob.glob("checkpoints/*_metrics.json")
    if not metric_files:
        print("No evaluation metric files (*_metrics.json) found in checkpoints/ directory.")
        return
        
    records = []
    for fpath in metric_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                
            # Extract run identifier/model name from filename or path
            fname = os.path.basename(fpath)
            model_name = fname.replace("_metrics.json", "")
            
            records.append({
                "Model / Checkpoint": model_name,
                "Accuracy (%)": f"{data.get('accuracy', 0)*100:.2f}%",
                "AUC-ROC": f"{data.get('auc_roc', 0):.4f}",
                "Precision": f"{data.get('precision', 0):.4f}",
                "Recall": f"{data.get('recall', 0):.4f}",
                "F1-Score": f"{data.get('f1_score', 0):.4f}",
                "Avg Guessing (g)": f"{data.get('avg_guessing', 0):.4f}",
                "Avg Slip (s)": f"{data.get('avg_slip', 0):.4f}"
            })
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            
    df = pd.DataFrame(records)
    
    print("\n" + "="*80)
    print("                      NEURAL CAT MODEL COMPARISON REPORT")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save the markdown comparison report
    report_path = "checkpoints/model_comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Neural CAT Models Comparison Report\n\n")
        f.write("This report aggregates evaluation results from all tested model checkpoints.\n\n")
        
        if records:
            headers = list(records[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in records:
                f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")
        
        f.write("\n\n> [!TIP]\n")
        f.write("> Run evaluation using `uv run python3 scripts/evaluate_neural_cat.py --checkpoint_path checkpoints/<model_checkpoint>.ckpt` to test new models and add them to this comparison.\n")
        
    print(f"\nSaved comparison table to Markdown report: {report_path}")

if __name__ == "__main__":
    main()
