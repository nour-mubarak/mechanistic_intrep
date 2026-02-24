# Literature Comparison & Novelty Assessment
## Cross-Lingual Mechanistic Bias (CLMB) Framework for Vision-Language Models

---

## 1. Literature Landscape

### 1.1 Sparse Autoencoders for Mechanistic Interpretability

| Work | Year | Domain | Scale | Key Contribution |
|---|---|---|---|---|
| **Cunningham et al.** "SAEs Find Highly Interpretable Features" | 2023 | LLMs (text-only) | Small models | First SAE → interpretable features in transformers |
| **Templeton et al. (Anthropic)** "Scaling Monosemanticity" | 2024 | Claude 3 Sonnet | 1M–34M features | Scaled SAEs to production LLM; found safety-relevant, multilingual, multimodal features |
| **Gao et al. (OpenAI)** "Scaling and Evaluating SAEs" | 2024 | GPT-4 | 16M latents | k-sparse SAEs; scaling laws; evaluation metrics |
| **Lieberum et al.** "Gemma Scope" | 2024 | Gemma 2 (2B, 9B, 27B) | Open-source SAE suite | JumpReLU SAEs; comprehensive layer coverage |
| **Olson et al.** "Hierarchical Structure in Vision Models" | 2025 | DINOv2 (vision-only) | Vision foundation model | SAEs on vision transformers; hierarchical taxonomy |
| **Marks et al.** "Sparse Feature Circuits" | 2024 | LLMs | Text-only | Causal circuits via SAE features; SHIFT method |
| **Tigges et al.** "Linear Representations of Sentiment" | 2023 | LLMs | Text-only | Linear probing + causal intervention for sentiment |
| **Our Work (CLMB)** | **2025** | **4 VLMs (3B–11B)** | **16K–32K features** | **First cross-lingual SAE analysis of VLMs for gender bias** |

### 1.2 Gender Bias in AI Models

| Work | Year | Domain | Method | Limitation vs. Ours |
|---|---|---|---|---|
| **Bolukbasi et al.** "Man is to Programmer..." | 2016 | Word embeddings | Geometric debiasing | Static embeddings only; no vision; no cross-lingual |
| **Zhao et al.** (various) | 2017–2019 | NLP models | Counterfactual data augmentation | Text-only; no mechanistic understanding |
| **Cho et al.** "DALL-Eval" | 2022 | Text-to-image | Behavioral evaluation | Measures output bias, not internal mechanism |
| **Birhane et al.** "Multimodal datasets" | 2021 | LAION-400M | Dataset audit | Documents bias in data, not model representations |
| **Our Work (CLMB)** | **2025** | **4 VLMs, EN+AR** | **SAE features + causal intervention** | **Mechanistic root cause + cross-lingual comparison** |

### 1.3 Cross-Lingual Bias Analysis

| Work | Year | Languages | Model Type | Limitation vs. Ours |
|---|---|---|---|---|
| **Lauscher & Glavaš** | 2019 | Multiple | Word embeddings | No vision; no SAEs; static representations |
| **Kaneko & Bollegala** | 2021 | English | BERT/GPT-2 | Single language only; no vision |
| **Stanczak et al.** | 2021 | Multiple | mBERT | Text-only; no mechanistic features |
| **Our Work (CLMB)** | **2025** | **Arabic + English** | **4 VLMs** | **First to use SAEs for cross-lingual VLM bias** |

---

## 2. Novelty Assessment: 8 Novel Contributions

### 🏆 Contribution 1: First SAE Application to Vision-Language Models for Bias Analysis
- **Gap:** All prior SAE work focuses on text-only LLMs (Anthropic, OpenAI, Gemma Scope) or vision-only models (Olson et al. 2025). No SAE work has been applied to multimodal VLMs for bias understanding.
- **Our contribution:** Train and evaluate SAEs across 4 VLMs (PaLiGemma-3B, Qwen2-VL-7B, LLaVA-1.5-7B, Llama-3.2-Vision-11B), achieving 71.7–99.9% explained variance.
- **Significance:** Demonstrates SAE feasibility for the VLM modality gap — language model layers in VLMs encode interpretable gender features despite receiving multimodal inputs.

### 🏆 Contribution 2: First Cross-Lingual SAE Feature Analysis
- **Gap:** Templeton et al. (2024) noted their SAE features are "multilingual" (same feature fires across languages), but never systematically compared feature sets between languages or measured alignment.
- **Our contribution:** We directly compare Arabic vs. English gender feature sets using Jaccard overlap, cosine similarity, and our novel CLBAS metric. Finding: **<1% feature overlap** across all 4 models.
- **Significance:** Provides the first empirical evidence that gender bias is encoded through **language-specific** neural mechanisms, not shared features — contradicting the assumption that multilingual models use shared representations.

### 🏆 Contribution 3: The CLMB Framework (4-Component System)
- **Gap:** No existing framework combines hierarchical bias localization, cross-lingual feature alignment, surgical intervention, and cross-lingual bias scoring.
- **Our contribution:** The CLMB framework with:
  - **HBL** (Hierarchical Bias Localization) — component-level attribution
  - **CLFA** (Cross-Lingual Feature Alignment) — optimal transport matching
  - **SBI** (Surgical Bias Intervention) — feature-level ablation with semantic preservation
  - **CLBAS** (Cross-Lingual Bias Alignment Score) — novel composite metric
- **Significance:** Provides a reusable, modular framework applicable to any multilingual model + language pair.

### 🏆 Contribution 4: CLBAS — A Novel Cross-Lingual Bias Metric
- **Gap:** No existing metric quantifies how similarly bias is encoded across languages at the feature level.
- **Our contribution:** CLBAS = Σ|bias(f_ar) − bias(f_en)| × sim(f_ar, f_en) / Σ sim(f_ar, f_en). Values near 0 → shared bias; values near 1 → language-specific bias.
- **Results:** CLBAS ranges from 0.0039 (Llama) to 0.1083 (PaLiGemma), all indicating language-specific encoding.
- **Significance:** Fills a measurement gap — enables quantitative comparison of bias structures across languages.

### 🏆 Contribution 5: Discovery of "Translation Amplification" Effect
- **Gap:** No prior work has documented how translation-based multilingual training amplifies gender markers.
- **Our finding:** PaLiGemma (translation-based) shows **Arabic > English** probe accuracy (88.6% vs. 85.3%), the only model with this inverted pattern. Root cause: Arabic has 1.44× more gender-marked words than English after translation.
- **Significance:** Reveals that translation pipelines inject additional gender signal, with practical implications for multilingual VLM training.

### 🏆 Contribution 6: Causal Intervention with Matched Random Control
- **Gap:** Most bias work uses correlational analysis (probes, embeddings). Templeton et al. (2024) perform feature steering but not systematic bias-targeted ablation. No prior work includes random ablation controls for SAE-based interventions.
- **Our contribution:** Targeted ablation of 100 gender features reduces gender terms by **19.2%**, while random ablation of 100 features only reduces by **7.5% ± 3.4%**. Effect specificity = 11.6 percentage points. Targeted is **2.5× more effective** than random.
- **Significance:** Provides the first causally-validated, controlled SAE-based debiasing experiment with appropriate baselines.

### 🏆 Contribution 7: 4-Model Comparative Architecture Study
- **Gap:** Prior SAE work focuses on single models. No study compares SAE properties and bias encoding across multiple VLM architectures.
- **Our contribution:** Systematic comparison across 4 VLMs spanning 3B–11B parameters, with different training regimes (translation-based, native multilingual, English-only fine-tuned).
- **Key finding:** Training regime determines bias pattern more than model size:
  - Native multilingual (Qwen, Llama) → balanced accuracy
  - English-only (LLaVA) → largest English bias (+6.4%)
  - Translation-based (PaLiGemma) → inverted pattern (Arabic +3.3%)
- **Significance:** First evidence that multilingual training strategy systematically shapes internal bias representation.

### 🏆 Contribution 8: Pronoun-Noun Differential Ablation Effect
- **Gap:** No prior work has examined how SAE feature ablation differentially affects different parts of speech.
- **Our finding:** Pronoun ablation is near-complete (he/his/him → 0, 100% reduction) while nouns are partially affected (woman −14%, man +24%). This reveals that SAE gender features primarily encode pronominal gender.
- **Significance:** Provides fine-grained understanding of what SAE "gender features" actually represent — critical for designing targeted debiasing interventions.

---

## 3. Detailed Comparison with Closest Related Work

### 3.1 vs. Templeton et al. 2024 (Anthropic — "Scaling Monosemanticity")

| Dimension | Templeton et al. | Our Work |
|---|---|---|
| **Model** | Claude 3 Sonnet (text-based LLM) | 4 VLMs (3B–11B, multimodal) |
| **SAE Scale** | 1M, 4M, 34M features | 16K–32K features |
| **Languages** | Multilingual features noted anecdotally | Systematic Arabic vs. English comparison |
| **Bias Analysis** | Found "gender bias awareness" feature (#34M/24442848); steering experiments | Full pipeline: probes → CLBAS → intervention + random control |
| **Cross-lingual** | Golden Gate Bridge fires in multiple languages (observation) | Quantified <1% feature overlap between languages |
| **Causal** | Feature steering on individual features | Systematic ablation of feature sets + matched random control |
| **Novelty gap we fill** | No cross-lingual bias quantification; no VLM analysis; no controlled debiasing experiment |

### 3.2 vs. Marks et al. 2024 ("Sparse Feature Circuits")

| Dimension | Marks et al. | Our Work |
|---|---|---|
| **Focus** | Circuit discovery in LLMs | Gender bias in VLMs |
| **Method** | SHIFT (ablate task-irrelevant features) | SBI (ablate bias-associated features) |
| **Evaluation** | Classifier generalization improvement | Gender term reduction + random control |
| **Cross-lingual** | Not addressed | Core contribution |
| **Novelty gap we fill** | No cross-lingual; no bias-specific application; no VLM analysis |

### 3.3 vs. Bolukbasi et al. 2016 ("Man is to Programmer")

| Dimension | Bolukbasi et al. | Our Work |
|---|---|---|
| **Representations** | Static word embeddings (Word2Vec) | Dynamic SAE features from VLM hidden states |
| **Method** | Geometric debiasing (projection) | Feature-level ablation during generation |
| **Scope** | English text only | Bilingual (Arabic + English), multimodal (image+text) |
| **Interpretability** | Gender direction in embedding space | Thousands of sparse, interpretable features |
| **Causal** | Post-hoc debiasing | During-generation intervention with control |
| **Novelty gap we fill** | Static → dynamic; text → multimodal; mono → cross-lingual; geometric → mechanistic |

---

## 4. Research Gap Analysis (Visual Summary)

```
EXISTING LITERATURE                          OUR CONTRIBUTION (FILLS GAP)
═══════════════════                          ═══════════════════════════

SAEs on text-only LLMs          ──────►      SAEs on 4 multimodal VLMs
  (Anthropic, OpenAI, Gemma)                   (PaLiGemma, Qwen, LLaVA, Llama)

Monolingual SAE analysis        ──────►      Cross-lingual (Arabic + English)
  (English features only)                      with <1% overlap finding

Behavioral bias measurement     ──────►      Mechanistic bias understanding
  (output statistics)                          (feature-level root cause)

Uncontrolled intervention       ──────►      Matched random ablation control
  (no baselines)                               (targeted 2.5× > random)

Single-model studies            ──────►      4-model architectural comparison
                                               (3B → 11B, 3 training regimes)

Gender direction (1D)           ──────►      Distributed features (1000s)
  (Bolukbasi et al.)                           with differential POS effects

No cross-lingual bias metric    ──────►      CLBAS composite metric
                                               (feature overlap + similarity + bias)
```

---

## 5. Positioning Statement

> **Our work is the first to apply sparse autoencoders to vision-language models for cross-lingual gender bias analysis.** We introduce the CLMB framework — a complete mechanistic interpretability pipeline that localizes, quantifies, and surgically intervenes on gender bias features across languages. Through systematic comparison of 4 VLMs (3B–11B), we provide the first empirical evidence that multilingual VLMs encode gender bias through **language-specific** features (<1% cross-lingual overlap), discover a novel "translation amplification" effect, and demonstrate causally-validated debiasing (19.2% reduction, 2.5× better than random ablation controls). This work bridges three previously disconnected research areas — sparse autoencoders, cross-lingual NLP, and VLM fairness — into a unified, reproducible framework.

---

## 6. Key Statistics for Publication Claims

| Claim | Evidence | Statistical Support |
|---|---|---|
| Gender features are language-specific | <1% overlap across all 4 models | Jaccard = 0.001–0.007 |
| Targeted ablation is not random | 19.2% vs. 7.5% reduction | Effect specificity = 11.6 pp |
| Translation amplifies gender | Arabic has 1.44× more gender words | Direct word count comparison |
| Probe gap is significant | EN−AR gap varies by training regime | McNemar χ²=50.05, p<0.0001, Cohen's d=2.25 |
| SAEs work on VLMs | 71.7–99.9% explained variance | 4-model replication |
| Pronouns are most affected | he/his/him → 0 (100% reduction) | Per-term ablation analysis |
| Training regime > model size | Pattern predicts bias direction | 4-model comparison |
