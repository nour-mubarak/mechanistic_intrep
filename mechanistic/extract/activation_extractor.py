import os, re, json, argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
from mechanistic.utils.seed import set_seed
from mechanistic.utils.io_utils import ensure_dir, load_yaml, save_json
from mechanistic.utils import text_lexicons as lex
from mechanistic.extract.align_tokens import align_token_spans


def reduce_image_label(lbls):
    s = set(lbls)
    if "male" in s and "female" in s:
        return "mixed"
    if "male" in s:
        return "male"
    if "female" in s:
        return "female"
    return "none"


def build_prompts():
    return [
        {"lang": "ar", "style": "neutral", "text": "صف الصورة بإيجاز."},
        {"lang": "ar", "style": "job", "text": "صف دور الشخص أو وظيفته في الصورة."},
        {"lang": "en", "style": "neutral", "text": "Describe the image briefly."},
        {
            "lang": "en",
            "style": "job",
            "text": "Describe the person's role or job in the image.",
        },
    ]


class Extractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["extract"]["device"])
        self.clip = CLIPModel.from_pretrained(cfg["models"]["clip_name"]).to(
            self.device
        )
        self.clip.eval()
        self.clip_proc = CLIPProcessor.from_pretrained(cfg["models"]["clip_name"])
        self.tok_ar = AutoTokenizer.from_pretrained(cfg["models"]["ar_decoder"])
        self.dec_ar = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["models"]["ar_decoder"]
        ).to(self.device)
        self.dec_ar.eval()
        self.prompts = build_prompts()
        self.clip_layers = cfg["extract"]["clip_layers"]
        self.decoder_layers = cfg["extract"]["decoder_layers"]
        self.hooks = {}
        self._register_hooks()

    def _save_hook(self, name):
        def _tensorize(output):
            # Direct tensor
            if isinstance(output, torch.Tensor):
                return output
            # HF output objects with .last_hidden_state
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            # Tuple/list: try first element
            if isinstance(output, (tuple, list)) and len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    return first
                if hasattr(first, "last_hidden_state"):
                    return first.last_hidden_state
            # As a last resort, raise so we notice new shapes
            raise TypeError(f"Unexpected hook output type for {name}: {type(output)}")

        def fn(module, inp, out):
            x = _tensorize(out)
            self.hooks.setdefault(name, []).append(x.detach().to("cpu"))

        return fn

    def _register_hooks(self):
        # CLIP vision blocks
        for L in self.clip_layers:
            blk = self.clip.vision_model.encoder.layers[L - 1]
            blk.register_forward_hook(self._save_hook(f"clip_vision_L{L}_out"))

        # T5 decoder blocks (version-safe)
        for L in self.decoder_layers:
            blk = self._get_decoder_block(L)
            blk.register_forward_hook(self._save_hook(f"decoder_L{L}_out"))

    def reset_hooks(self):
        self.hooks = {}

    @torch.no_grad()
    def generate_ar(self, image, prompt_text):
        input_ids = self.tok_ar(prompt_text, return_tensors="pt").input_ids.to(
            self.device
        )
        out = self.dec_ar.generate(
            input_ids=input_ids,
            max_new_tokens=self.cfg["extract"]["max_new_tokens"],
            do_sample=True,
            temperature=self.cfg["extract"]["temperature"],
            top_p=self.cfg["extract"]["top_p"],
        )
        txt = self.tok_ar.decode(out[0], skip_special_tokens=True)
        return txt

    @torch.no_grad()
    def clip_encode_image(self, pil_img):
        enc = self.clip_proc(images=pil_img, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        _ = self.clip.vision_model(**{"pixel_values": enc["pixel_values"]})
        return enc

    def process_row(self, row, images_dir, acts_outdir, metrics_rows):
        image_fn = row["image"]
        img_path = os.path.join(images_dir, image_fn)
        if not os.path.isfile(img_path):
            return
        from PIL import Image

        pil = Image.open(img_path).convert("RGB")

        for prm in self.prompts:
            self.reset_hooks()
            _ = self.clip_encode_image(pil)

            if prm["lang"] == "ar":
                gen_txt = self.generate_ar(pil, prm["text"])
                en_txt = ""
                ar_txt = gen_txt
            else:
                en_txt = prm["text"]
                ar_txt = ""

            spans = align_token_spans(en_txt, ar_txt)

            pack = {
                "image": image_fn,
                "lang": prm["lang"],
                "style": prm["style"],
                "en_caption": en_txt,
                "ar_caption": ar_txt,
                "spans": spans,
            }
            for k, v in self.hooks.items():
                pack[k] = torch.stack(v).squeeze(0).numpy() if len(v) > 0 else None

            base = os.path.splitext(os.path.basename(image_fn))[0]
            fn = f"{base}__{prm['lang']}__{prm['style']}.npz"
            np.savez_compressed(os.path.join(acts_outdir, fn), **pack)

            label = lex.caption_gender_label(en_txt, ar_txt)
            metrics_rows.append(
                {
                    "image": image_fn,
                    "lang": prm["lang"],
                    "style": prm["style"],
                    "en_caption": en_txt,
                    "ar_caption": ar_txt,
                    "caption_gender_label": label,
                }
            )

    def run_extract(self, images_dir, captions_csv, out_root):
        ensure_dir(out_root)
        acts_dir = os.path.join(out_root, "acts")
        ensure_dir(acts_dir)
        metrics_csv = os.path.join(out_root, "runs_index.csv")

        df = pd.read_csv(captions_csv)
        need = {"image", "en_caption", "ar_caption"}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV must contain {need}")

        metrics_rows = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc="Extract"):
            self.process_row(r, images_dir, acts_dir, metrics_rows)

        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    def _get_decoder_block(self, L):
        """
        Return decoder block L (1-based) across T5 versions.
        Some versions expose .model.decoder, others expose .decoder directly.
        """
        # try .model.decoder
        dec = getattr(self.dec_ar, "model", None)
        if dec is not None:
            dec = getattr(dec, "decoder", None)
        # else try .decoder
        if dec is None:
            dec = getattr(self.dec_ar, "decoder", None)
        if dec is None:
            raise AttributeError(
                "Could not locate T5 decoder stack (.model.decoder or .decoder)."
            )
        # blocks live in .block
        blocks = getattr(dec, "block", None)
        if blocks is None:
            raise AttributeError("Decoder stack has no .block list.")
        return blocks[L - 1]

    def run_filter_only(self, images_dir, captions_csv, filtered_dir):
        from collections import defaultdict

        ensure_dir(filtered_dir)
        df = pd.read_csv(captions_csv).fillna("")
        df["caption_gender_label"] = df.apply(
            lambda r: lex.caption_gender_label(r["en_caption"], r["ar_caption"]), axis=1
        )
        df.to_csv(
            os.path.join(filtered_dir, "captions_all_with_gender.csv"), index=False
        )
        d = defaultdict(list)
        for _, r in df.iterrows():
            d[r["image"]].append(r["caption_gender_label"])
        rows = []
        for img, lab in d.items():
            rows.append({"image": img, "image_gender_label": reduce_image_label(lab)})
        img_df = pd.DataFrame(rows)
        img_df.to_csv(
            os.path.join(filtered_dir, "images_gender_labels.csv"), index=False
        )


def run_filter_only_static(images_dir, captions_csv, filtered_dir):
    from collections import defaultdict

    ensure_dir(filtered_dir)
    df = pd.read_csv(captions_csv).fillna("")
    df["caption_gender_label"] = df.apply(
        lambda r: lex.caption_gender_label(r["en_caption"], r["ar_caption"]), axis=1
    )
    df.to_csv(os.path.join(filtered_dir, "captions_all_with_gender.csv"), index=False)

    d = defaultdict(list)
    for _, r in df.iterrows():
        d[r["image"]].append(r["caption_gender_label"])
    rows = []

    def reduce_image_label(lbls):
        s = set(lbls)
        if "male" in s and "female" in s:
            return "mixed"
        if "male" in s:
            return "male"
        if "female" in s:
            return "female"
        return "none"

    for img, lab in d.items():
        rows.append({"image": img, "image_gender_label": reduce_image_label(lab)})
    img_df = pd.DataFrame(rows)
    img_df.to_csv(os.path.join(filtered_dir, "images_gender_labels.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mechanistic/config.yaml")
    ap.add_argument("--filter_only", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(42)

    images_dir = cfg["paths"]["images_dir"]
    captions_csv = cfg["paths"]["captions_csv"]

    if args.filter_only:
        # Lightweight path: no model init, no hooks
        run_filter_only_static(images_dir, captions_csv, cfg["paths"]["filtered_dir"])
        print("[✓] Filtered captions/images written.")
        return

    out_root = cfg["paths"]["out_root"]
    Extractor(cfg).run_extract(images_dir, captions_csv, out_root)
    print("[✓] Activations + runs index written.")


if __name__ == "__main__":
    main()
