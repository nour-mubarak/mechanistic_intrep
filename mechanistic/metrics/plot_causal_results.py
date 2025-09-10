import os, argparse, pandas as pd, matplotlib.pyplot as plt
from mechanistic.utils.io_utils import load_yaml, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    ap.add_argument("--before_csv", required=True)
    ap.add_argument("--after_csv", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = cfg["paths"]["figs_dir"]; ensure_dir(out_dir)

    dfb = pd.read_csv(args.before_csv)
    dfa = pd.read_csv(args.after_csv)

    def summarize(df):
        return {
            "GMR": df["GMR"].mean(),
            "OAR": df["OAR"].mean(),
            "SAR": df["SAR"].mean(),
            "RR":  df["RR"].mean(),
            "clip_sim": df["clip_sim"].mean()
        }

    sb = summarize(dfb); sa = summarize(dfa)
    keys = list(sb.keys())
    b = [sb[k] for k in keys]; a = [sa[k] for k in keys]

    plt.figure()
    x = range(len(keys))
    plt.bar(x, b, label="before")
    plt.bar([i+0.4 for i in x], a, width=0.4, label="after")
    plt.xticks([i+0.2 for i in x], keys)
    plt.title("Causal Patch: Δ metrics")
    plt.legend()
    p = os.path.join(out_dir, "causal_bar.png")
    plt.savefig(p, bbox_inches="tight", dpi=160)
    print("[✓] Saved", p)

if __name__ == "__main__":
    main()
