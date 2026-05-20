import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)


def plot_summary_results():
    df = pd.read_csv("results/summary_results.csv")

    metrics = ["exact_match", "retrieval_hit", "avg_score"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(df["method"], df[metric])
        plt.title(f"{metric} comparison between methods")
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(f"results/figures/{metric}_comparison.png", dpi=300)
        plt.close()


def plot_type_comparison():
    df = pd.read_csv("results/type_comparison_results.csv")

    for metric in ["exact_match", "retrieval_hit", "score"]:
        pivot_df = df.pivot(index="method", columns="type", values=metric)

        pivot_df.plot(kind="bar", figsize=(8, 5))
        plt.title(f"{metric} by question type")
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(f"results/figures/{metric}_by_question_type.png", dpi=300)
        plt.close()


def plot_self_evolving_rounds():
    path = "results/self_evolving_results.csv"

    if not os.path.exists(path):
        print("self_evolving_results.csv not found.")
        return

    df = pd.read_csv(path)

    if "round" not in df.columns or "auto_score" not in df.columns:
        print("Missing round or auto_score column.")
        return

    round_df = df.groupby("round")["auto_score"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(round_df["round"], round_df["auto_score"], marker="o")
    plt.title("Auto-score across self-evolving rounds")
    plt.xlabel("Round")
    plt.ylabel("Average Auto-score")
    plt.tight_layout()
    plt.savefig("results/figures/self_evolving_rounds.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_summary_results()
    plot_type_comparison()
    plot_self_evolving_rounds()

    print("Done. Figures saved to results/figures/")