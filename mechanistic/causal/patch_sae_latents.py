import os, argparse, numpy as np, torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mechanistic.utils.io_utils import load_yaml
from mechanistic.utils.seed import set_seed
import pandas as pd

class SAE(nn.Module):
    def __init__(self, d_in, k_latents):
        super().__init__()
        self.enc = nn.Linear(d_in, k_latents, bias=True)
        self.dec = nn.Linear(k_latents, d_in, bias=True)
        self.act = nn.ReLU()
    def encode(self, x): return self.act(self.enc(x))
    def decode(self, z): return self.dec(z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    ap.add_argument("--layer", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(42)
    layer = args.layer or cfg["sae"]["layer"]
    acts_dir = cfg["paths"].get("activations_dir", os.path.join(cfg["paths"]["out_root"],"acts"))
    sae_dir = cfg["paths"]["sae_dir"]
    k = cfg["sae"]["k_latents"]
    sae_path = os.path.join(sae_dir, f"sae_{layer}_k{k}.pt")
    top_csv = os.path.join(sae_dir, "gender_top_features.csv")
    top = pd.read_csv(top_csv)["feature"].tolist()[:10]

    files = [f for f in os.listdir(acts_dir) if f.endswith(".npz")]
    if not files:
        raise RuntimeError("No activation files found.")
    pack = np.load(os.path.join(acts_dir, files[0]), allow_pickle=True)
    A = pack[f"{layer}_out"]
    if A.ndim==3: A=A[0]
    T,H = A.shape
    lo = max(0, T//2 - cfg["causal"]["window_tokens"]); hi = min(T, T//2 + cfg["causal"]["window_tokens"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SAE(d_in=H, k_latents=k).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    with torch.no_grad():
        X = torch.tensor(A[lo:hi]).float().to(device)
        Z = sae.encode(X)
        Z[:, top] = Z[:, top] * cfg["causal"]["latent_scale"]
        Xp = sae.decode(Z).cpu().numpy()

    tok = AutoTokenizer.from_pretrained(cfg["models"]["ar_decoder"])
    dec = AutoModelForSeq2SeqLM.from_pretrained(cfg["models"]["ar_decoder"]).to(device).eval()
    blk_idx = int(layer.split("_")[1]) - 1
    block = dec.model.decoder.block[blk_idx]

    def pre_hook(module, inputs):
        hidden = inputs[0]
        hidden[:, lo:hi, :] = torch.tensor(Xp).to(hidden.device).to(hidden.dtype)
        return (hidden,) + inputs[1:]

    h = block.register_forward_pre_hook(pre_hook)
    ids = tok("صف الصورة بإيجاز.", return_tensors="pt").input_ids.to(device)
    out = dec.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9)
    txt = tok.decode(out[0], skip_special_tokens=True)
    h.remove()
    print("[latent-patched caption]", txt)

if __name__ == "__main__":
    main()
