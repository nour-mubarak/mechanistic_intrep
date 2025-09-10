import os, argparse, numpy as np, pandas as pd, torch
from torch import nn
from mechanistic.utils.io_utils import load_yaml

class SAE(nn.Module):
    def __init__(self, d_in, k_latents):
        super().__init__()
        self.enc = nn.Linear(d_in, k_latents, bias=True)
        self.dec = nn.Linear(k_latents, d_in, bias=True)
        self.act = nn.ReLU()
    def forward(self, x):
        z = self.act(self.enc(x))
        x_hat = self.dec(z)
        return x_hat, z

def collect_acts(acts_dir, layer_key):
    X = []
    y = []
    meta = []
    for fn in os.listdir(acts_dir):
        if not fn.endswith(".npz"): continue
        pack = np.load(os.path.join(acts_dir, fn), allow_pickle=True)
        spans = pack["spans"].item()
        arr_key = f"{layer_key}_out" if not layer_key.endswith("_out") else layer_key
        if arr_key not in pack or pack[arr_key] is None: continue
        A = pack[arr_key]
        if A.ndim == 3: A = A[0]
        T,H = A.shape
        label = 1 if (len(spans["male"]) or len(spans["female"])) else 0
        c = T//2; lo=max(0,c-4); hi=min(T,c+4)
        X.append(A[lo:hi].reshape(-1,H))
        y.extend([label]* (hi-lo))
        meta.append((fn,label))
    if not X: return None, None, None
    return np.concatenate(X,axis=0), np.array(y), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    acts_dir = cfg["paths"].get("activations_dir", os.path.join(cfg["paths"]["out_root"],"acts"))
    layer = cfg["sae"]["layer"]
    k = cfg["sae"]["k_latents"]
    sae_path = os.path.join(cfg["paths"]["sae_dir"], f"sae_{layer}_k{k}.pt")

    X, y, meta = collect_acts(acts_dir, layer)
    if X is None: raise RuntimeError("No activations found.")
    H = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAE(d_in=H, k_latents=k).to(device)
    model.load_state_dict(torch.load(sae_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = model.act(model.enc(torch.tensor(X).float().to(device))).cpu().numpy()

    sel = z[y==1].mean(0) - z[y==0].mean(0)
    order = np.argsort(-sel)
    top = order[:50]
    df = pd.DataFrame({"feature": top, "selectivity": sel[top]})
    out_csv = os.path.join(cfg["paths"]["sae_dir"], "gender_top_features.csv")
    df.to_csv(out_csv, index=False)
    print("[✓] Wrote", out_csv)

if __name__ == "__main__":
    main()
