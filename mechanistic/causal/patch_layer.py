import os, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mechanistic.utils.io_utils import load_yaml
from mechanistic.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    ap.add_argument("--layer", default=None, help="e.g., decoder_8")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(42)

    layer = args.layer or cfg["sae"]["layer"]
    acts_dir = cfg["paths"].get(
        "activations_dir", os.path.join(cfg["paths"]["out_root"], "acts")
    )
    files = [f for f in os.listdir(acts_dir) if f.endswith(".npz")]
    if not files:
        raise RuntimeError("No activations npz found.")

    tok = AutoTokenizer.from_pretrained(cfg["models"]["ar_decoder"])
    dec = AutoModelForSeq2SeqLM.from_pretrained(cfg["models"]["ar_decoder"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = dec.to(device).eval()
    target_block_idx = int(layer.split("_")[1]) - 1

    def get_decoder_block(model, idx0):
        dec = getattr(model, "model", None)
        if dec is not None:
            dec = getattr(dec, "decoder", None)
        if dec is None:
            dec = getattr(model, "decoder", None)
        if dec is None:
            raise AttributeError(
                "Could not locate T5 decoder (.model.decoder or .decoder)."
            )
        blocks = getattr(dec, "block", None)
        if blocks is None:
            raise AttributeError("Decoder has no .block.")
        return blocks[idx0]

    blk_idx = int(layer.split("_")[1]) - 1
    block = get_decoder_block(dec, blk_idx)

    src_pack = np.load(os.path.join(acts_dir, files[0]), allow_pickle=True)
    base_pack = np.load(os.path.join(acts_dir, files[-1]), allow_pickle=True)
    src_A = src_pack[f"{layer}_out"]
    base_A = base_pack[f"{layer}_out"]
    if src_A.ndim == 3:
        src_A = src_A[0]
    if base_A.ndim == 3:
        base_A = base_A[0]
    T, H = src_A.shape
    lo = max(0, T // 2 - 3)
    hi = min(T, T // 2 + 3)

    def pre_hook(module, inputs):
        hidden = inputs[0]
        hidden[:, lo:hi, :] = (
            torch.tensor(base_A[lo:hi]).to(hidden.device).to(hidden.dtype)
        )
        return (hidden,) + inputs[1:]

    h = block.register_forward_pre_hook(pre_hook)
    prompt = "صف الصورة بإيجاز."
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = dec.generate(
        input_ids=ids, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    h.remove()

    print("[patched caption]", txt)


if __name__ == "__main__":
    main()
