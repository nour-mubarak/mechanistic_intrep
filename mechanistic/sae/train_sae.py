import os, json, argparse, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from mechanistic.utils.io_utils import load_yaml, ensure_dir
from mechanistic.utils.seed import set_seed

class SAEDataset(Dataset):
    def __init__(self, acts_dir, layer_key, window_tokens=6):
        self.X = []
        for fn in os.listdir(acts_dir):
            if not fn.endswith(".npz"): continue
            pack = np.load(os.path.join(acts_dir, fn), allow_pickle=True)
            spans = pack["spans"].item()
            arr_key = f"{layer_key}_out" if not layer_key.endswith("_out") else layer_key
            if arr_key not in pack: continue
            A = pack[arr_key]
            if A is None: continue
            if A.ndim == 2:
                T, H = A.shape
            elif A.ndim == 3:
                A = A[0]
                T, H = A.shape
            else:
                continue
            centers = []
            if len(spans["male"]) or len(spans["female"]):
                centers.append(T//2)
            else:
                continue
            for c in centers:
                lo = max(0, c - window_tokens)
                hi = min(T, c + window_tokens + 1)
                self.X.append(A[lo:hi].reshape(-1, H))
        if self.X:
            self.X = np.concatenate(self.X, axis=0)
        else:
            self.X = np.zeros((0, 1024), dtype=np.float32)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx].astype(np.float32)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(42)

    acts_dir = cfg["paths"].get("activations_dir", os.path.join(cfg["paths"]["out_root"],"acts"))
    out_dir  = cfg["paths"]["sae_dir"]
    ensure_dir(out_dir)

    layer = cfg["sae"]["layer"]
    k = cfg["sae"]["k_latents"]
    lam = cfg["sae"]["l1_lambda"]
    bs = cfg["sae"]["batch_size"]
    lr = cfg["sae"]["lr"]
    epochs = cfg["sae"]["epochs"]

    ds = SAEDataset(acts_dir, layer_key=layer, window_tokens=cfg.get("causal",{}).get("window_tokens",6))
    if len(ds) == 0:
        raise RuntimeError("No activation windows collected. Re-check acts dir and layer key.")
    H = ds[0].shape[-1]
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAE(d_in=H, k_latents=k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"SAE epoch {ep+1}/{epochs}")
        for x in pbar:
            x = x.to(device)
            x_hat, z = model(x)
            loss = mse(x_hat, x) + lam * z.abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
    torch.save(model.state_dict(), os.path.join(out_dir, f"sae_{layer}_k{k}.pt"))
    print("[✓] SAE saved.")

if __name__ == "__main__":
    main()
