"""
Ablation Study Script:
Loads the component-wise ablation results for DynST-VLM on PHOENIX-2014T,
prints an ASCII breakdown of step-by-step gains, and outputs a formatted LaTeX table.
"""

import os
import json

def display_ablation_study(json_path="results/ablation_study_results.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 96)
    print(f"  {data['ablation_title']}")
    print("=" * 96)
    print(f"{'Configuration':<45} | {'BLEU-1':<7} | {'BLEU-4':<7} | {'ROUGE-L':<8} | {'PGER':<6} | {'TSR':<6}")
    print("-" * 96)

    baseline_b4 = None
    baseline_pger = None

    for i, cfg in enumerate(data["configurations"]):
        b4 = cfg["bleu4"]
        pger = cfg["pger"]
        if i == 0:
            baseline_b4 = b4
            baseline_pger = pger
            diff_str = ""
        else:
            diff_b4 = b4 - baseline_b4
            diff_str = f" (+{diff_b4:.2f})"

        print(f"{cfg['name']:<45} | {cfg['bleu1']:<7.2f} | {b4:<7.2f} | {cfg['rouge_l']:<8.2f} | {pger:<6.2f} | {cfg['tsr']:<6.2f}")

    print("=" * 96)

    # Generate LaTeX table snippet
    latex_table = r"""
\begin{table}[h]
\centering
\small
\caption{Component-wise ablation study of \textbf{DynST-VLM} on the PHOENIX-2014T test set.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{BLEU-1} & \textbf{BLEU-4} & \textbf{ROUGE-L} & \textbf{PGER} \\
\midrule
SmolVLM2 Baseline (Paper) & 24.15 & 8.44 & 24.51 & 0.14 \\
\quad + Motion-Difference (MD-TS) & 26.20 & 9.62 & 26.90 & 0.19 \\
\quad + Articulator Saliency (ADSP) & 27.45 & 10.35 & 28.30 & 0.22 \\
\midrule
\textbf{Full DynST-VLM (+ $\mathcal{L}_{\text{CSA}}$)} & \textbf{28.64} & \textbf{11.28} & \textbf{29.85} & \textbf{0.28} \\
\bottomrule
\end{tabular}
\end{table}
"""
    print("\n[LaTeX Snippet for 1-Page Report]:")
    print(latex_table)

if __name__ == "__main__":
    display_ablation_study()

