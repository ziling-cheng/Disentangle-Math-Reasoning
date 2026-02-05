import pandas as pd
import numpy as np
from argparse import ArgumentParser

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paths import RESULT_DIR, BEHAVIOUR_PLOT_DIR

def sort_rows_by_model_size(df):
    size = df["model"].str.extract(r"-(\d+(?:\.\d+)?)B-", expand=False).astype(float)
    df_sorted = df.assign(_sizeB=size).sort_values("_sizeB").drop(columns="_sizeB")
    return df_sorted

def collect_accuracies_by_settings(df):
    accuracy = []
    for _, row in df.iterrows():
        accuracy.append(row['original_zs_acc']*100)
        accuracy.append(row['arithmetic_computation_zs_acc']*100)
        accuracy.append(row['numerical_abstraction_zs_acc']*100)
        accuracy.append(row['symbolic_abstraction_zs_acc']*100)
    return accuracy

def summarize_zs_behaviour_results(df, dataset_name):
    
    llama_df = sort_rows_by_model_size(results[results['model'].str.contains('Llama')])
    qwen_df = sort_rows_by_model_size(results[results['model'].str.contains('Qwen')])
    
    qwen_accuracy = collect_accuracies_by_settings(qwen_df)
    llama_accuracy = collect_accuracies_by_settings(llama_df)

    qwen_df = pd.DataFrame({
        'Model Family': ['Qwen2.5'] * 16,
        'Model Size': ['3B', '3B','3B','3B', '7B', '7B', '7B', '7B', '14B', '14B', '14B', '14B', '32B', '32B', '32B', '32B'],
        'Setting': ['Original', 'Arith. Computation', "Numerical Abstr.", 'Symbolic Abstr.'] * 4,
        'Accuracy': qwen_accuracy
    })
    qwen_df['Model'] = qwen_df['Model Family'] + '-' + qwen_df['Model Size']

    llama_df = pd.DataFrame({
        'Model Family': ['Llama-3'] * 12,
        'Model Size': ['1B', '1B', '1B', '1B', '3B', '3B', '3B','3B', '8B', '8B', '8B', '8B'],
        'Setting': ['Original', 'Arith. Computation', "Numerical Abstr.", 'Symbolic Abstr.'] * 3,
        'Accuracy': llama_accuracy
    })
    llama_df['Model'] = llama_df['Model Family'] + '-' + llama_df['Model Size']

    # Combine and plot
    combined_df = pd.concat([ llama_df, qwen_df], ignore_index=True)

    plt.figure(figsize=(18, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)

    ax = sns.barplot(
        data=combined_df,
        x="Model", y="Accuracy", hue="Setting",
        palette="Set2", width=0.8,
        hue_order=["Original", "Arith. Computation", "Numerical Abstr.", "Symbolic Abstr."]
    )

    ax.set_xlabel("")  # Remove x-axis label
    ax.set_ylabel("Accuracy", fontsize=22)  # Set y-axis label and font size


    # Add score labels to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=0, fontsize=16)

    ax.legend(fontsize=18)



    ax.set_title(f"Zero-Shot Without CoT Accuracy on {dataset_name.upper()} by Model Family and Size", fontsize=22, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), ha='center', rotation=0, fontsize=18)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"{BEHAVIOUR_PLOT_DIR}/zs_results_{dataset_name}.png", dpi=300, bbox_inches='tight')
    #plt.show()

def make_cot_comparison_per_model(no_cot, cot, model_name, dataset_name):
    groups = [
        "Original (Q → 8)",
        "Arith. Comp. (5+3 → 8)",
        "Num. Abstr. (Q → 5+3)",
        "Sym. Abstr. (Q → x+y)",
    ]

    base_colors = ["#F6D77A", "#B79AE8", "#3F62D6", "#3FA3A2"]

    def lighten(hex_color, amount=0.60):
        import matplotlib.colors as mc
        c = np.array(mc.to_rgb(hex_color))
        return tuple((1 - amount) * c + amount * np.ones(3))

    light_colors = [lighten(c, 0.60) for c in base_colors]
    dark_colors  = [tuple(np.array(plt.matplotlib.colors.to_rgb(c))) for c in base_colors]


    x = np.arange(len(groups)) * 1.35
    w = 0.34

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)

    bars_no = ax.bar(x - w/2, no_cot, width=w, color=light_colors, edgecolor="none")
    bars_c  = ax.bar(x + w/2, cot,    width=w, color=dark_colors,  edgecolor="none")

    # value labels
    for bars in (bars_no, bars_c):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    # y axis / grid
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", alpha=0.20)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # x tick labels: repeated No-CoT / CoT under each bar
    xticks, xlabels = [], []
    for xi in x:
        xticks += [xi - w/2, xi + w/2]
        xlabels += ["No CoT", "CoT"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    # ---- legend: one entry per evaluation setting (2x2), above the axes (compact)
    group_handles = [Patch(facecolor=base_colors[i], edgecolor="none", label=groups[i])
                     for i in range(len(groups))]
    fig.legend(handles=group_handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.98))

    # tight margins like a paper figure
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.08, right=0.98)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    
    plt.savefig(f"{BEHAVIOUR_PLOT_DIR}/{model_name}_cot_improvement_{dataset_name}.png", dpi=300, bbox_inches='tight')
    #plt.show()

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Dataset name",
    )
    args = parser.parse_args()
    
    results = pd.read_csv( f"{RESULT_DIR}/{args.dataset_name}_evaluation_summary.csv")
    results = results.rename(columns={"Unnamed: 0": "model"})
    
    # one plot for zs results for all models
    summarize_zs_behaviour_results(df=results, dataset_name=args.dataset_name)
    
    # one plot comparing cot vs. no cot performance per model
    for _, row in results.iterrows():
        no_cot = []
        cot = []

        for eval_setting in ['original_{}_acc', 'arithmetic_computation_{}_acc', 'numerical_abstraction_{}_acc', 'symbolic_abstraction_{}_acc']:
            no_cot.append(row[eval_setting.format("zs")]*100)
            cot.append(row[eval_setting.format("cot")]*100)
        make_cot_comparison_per_model(no_cot, cot, row['model'], args.dataset_name)
    
    