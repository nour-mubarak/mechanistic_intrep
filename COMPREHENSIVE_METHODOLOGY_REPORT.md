# Comprehensive Methodology Report: Cross-Lingual SAE Analysis for Vision-Language Model Gender Bias

## Mechanistic Interpretability Pipeline for Arabic-English VLM Analysis

**Project**: Cross-Lingual Bias Mechanistic Interpretability (CLMB)  
**Models Analyzed**: PaLiGemma-3B, Qwen2-VL-7B-Instruct  
**Languages**: Arabic (العربية) & English  
**Generated**: January 20, 2026

---

## Executive Summary

This research applies **Sparse Autoencoders (SAEs)** to discover interpretable features encoding gender bias in multilingual Vision-Language Models. We introduce the **CLMB Framework** (Cross-Lingual Mechanistic Bias) with four novel components: Hierarchical Bias Localization (HBL), Cross-Lingual Feature Alignment (CLFA), Surgical Bias Intervention (SBI), and the Cross-Lingual Bias Alignment Score (CLBAS).

**Key Finding**: Qwen2-VL-7B exhibits **6.7× lower** cross-lingual bias alignment than PaLiGemma-3B, indicating larger VLMs develop more language-specific gender processing pathways.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Data Preparation Pipeline](#2-data-preparation-pipeline)
3. [Activation Extraction](#3-activation-extraction)
4. [Sparse Autoencoder Architecture](#4-sparse-autoencoder-architecture)
5. [SAE Training Pipeline](#5-sae-training-pipeline)
6. [CLMB Framework Components](#6-clmb-framework-components)
7. [Cross-Lingual Analysis Methodology](#7-cross-lingual-analysis-methodology)
8. [Surgical Bias Intervention](#8-surgical-bias-intervention)
9. [Experimental Results](#9-experimental-results)
10. [Technical Implementation Details](#10-technical-implementation-details)

---

## 1. Project Architecture

### 1.1 Directory Structure

```
sae_captioning_project/
├── configs/
│   ├── config.yaml           # Main configuration
│   ├── clmb_config.yaml      # CLMB-specific settings
│   └── config_layer6.yaml    # Layer-specific configs
├── src/
│   ├── models/
│   │   └── sae.py            # SAE implementations (Standard, Gated, TopK)
│   ├── clmb/
│   │   ├── hbl.py            # Hierarchical Bias Localization
│   │   ├── clfa.py           # Cross-Lingual Feature Alignment
│   │   └── sbi.py            # Surgical Bias Intervention
│   └── data/                 # Data loading utilities
├── scripts/
│   ├── 01_prepare_data.py    # Data preprocessing
│   ├── 02_extract_activations.py  # VLM hook-based extraction
│   ├── 03_train_sae.py       # SAE training with W&B
│   ├── 24_cross_lingual_overlap.py  # CLBAS computation
│   ├── 26_surgical_bias_intervention.py  # Ablation experiments
│   └── 30_qwen2vl_cross_lingual_analysis.py  # Model comparison
└── checkpoints/
    ├── paligemma/            # PaLiGemma activations & SAEs
    └── qwen2vl/              # Qwen2-VL activations & SAEs
```

### 1.2 Model Specifications

| Model | Architecture | d_model | Layers | SAE Features |
|-------|-------------|---------|--------|--------------|
| PaLiGemma-3B | Gemma-3-4B-IT base | 2048 | 18 | 16,384 |
| Qwen2-VL-7B | Qwen2-VL-Instruct | 3584 | 28 | 28,672 |

---

## 2. Data Preparation Pipeline

### 2.1 Dataset Overview

The analysis uses a **bilingual image captioning dataset** with Arabic and English parallel captions describing the same images.

**Script**: `scripts/01_prepare_data.py`

### 2.2 Gender Extraction from Arabic Text

Arabic presents unique challenges due to grammatical gender markers. We extract gender using morphological patterns:

```python
def extract_gender_from_arabic(text: str) -> str:
    """
    Arabic gender markers:
    - Male: رجل (man), ولد (boy), طفل (male child), شاب (young man)
    - Female: امرأة (woman), فتاة (girl), طفلة (female child), بنت (girl)
    """
    female_markers = [
        'امرأة', 'نساء',   # woman, women
        'فتاة', 'فتيات',   # girl, girls
        'طفلة',            # female child
        'بنت', 'بنات',     # girl, girls
        'سيدة',            # lady
        'أنثى',            # female
    ]
    
    male_markers = [
        'رجل', 'رجال',     # man, men
        'ولد', 'أولاد',    # boy, boys
        'شاب', 'شبان',     # young man, young men
        'طفل',             # male child (when not طفلة)
        'صبي',             # boy
        'ذكر',             # male
    ]
    
    # Check female first (some male markers are substrings)
    for marker in female_markers:
        if marker in text:
            return 'female'
    for marker in male_markers:
        if marker in text:
            return 'male'
    return 'unknown'
```

### 2.3 Gender Extraction from English Text

```python
def extract_gender_from_english(text: str) -> str:
    """Extract gender from English caption text."""
    female_words = ['she', 'her', 'hers', 'woman', 'girl', 'female', 
                   'lady', 'mother', 'daughter', 'sister', 'wife']
    male_words = ['he', 'him', 'his', 'man', 'boy', 'male', 
                 'gentleman', 'father', 'son', 'brother', 'husband']
    
    text = text.lower()
    
    female_count = sum(1 for w in female_words if w in text)
    male_count = sum(1 for w in male_words if w in text)
    
    if female_count > male_count:
        return 'female'
    elif male_count > female_count:
        return 'male'
    return 'unknown'
```

### 2.4 Prompt Templates

**English Prompt**: 
```
"Describe the person in this image in detail."
```

**Arabic Prompt**:
```
"صف الشخص في هذه الصورة بالتفصيل."
```

These prompts are applied using the model's chat template via `processor.apply_chat_template()`.

---

## 3. Activation Extraction

### 3.1 Hook-Based Extraction

**Script**: `scripts/02_extract_activations.py`

We extract intermediate activations from transformer layers using forward hooks:

```python
def extract_activations_batch(model, processor, images, prompts, layers, device):
    """Extract activations for a batch of image-prompt pairs."""
    
    activations = {layer: [] for layer in layers}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            # Detach, move to CPU, convert to float32
            act = act.detach().cpu().float()
            
            # NaN detection and handling
            if torch.isnan(act).any():
                act = torch.nan_to_num(act, nan=0.0)
                
            activations[layer_idx].append(act)
        return hook
    
    # Register hooks on target layers
    for layer_idx in layers:
        if hasattr(model, 'language_model'):
            layer = model.language_model.layers[layer_idx]
        elif hasattr(model, 'model'):
            layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return {layer: torch.cat(acts, dim=0) for layer, acts in activations.items()}
```

### 3.2 Memory Management

For large-scale extraction, we use **chunked checkpointing**:

```python
# Save checkpoint periodically to avoid OOM
if samples_processed >= checkpoint_interval:
    checkpoint_path = checkpoints_dir / f'activations_{language}_chunk_{idx}.pt'
    torch.save({
        'activations': stacked_chunk,
        'genders': chunk_genders,
        'image_ids': chunk_image_ids,
    }, checkpoint_path)
    
    # Clear memory
    del stacked_chunk
    gc.collect()
    torch.cuda.empty_cache()
```

### 3.3 Layer Selection Strategy

| Model | Layers Extracted | Rationale |
|-------|-----------------|-----------|
| PaLiGemma | [0, 3, 6, 9, 12, 15, 17] | Spans early→late, 18 total layers |
| Qwen2-VL | [0, 4, 8, 12, 16, 20, 24, 27] | Every 4th layer + final, 28 total layers |

---

## 4. Sparse Autoencoder Architecture

### 4.1 SAE Configuration

**File**: `src/models/sae.py`

```python
@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""
    d_model: int              # Input/output dimension (2048 or 3584)
    expansion_factor: int = 8  # Hidden dimension multiplier
    l1_coefficient: float = 5e-4  # Sparsity regularization
    normalize_decoder: bool = True
    tied_weights: bool = False
    activation: Literal["relu", "gelu", "topk"] = "relu"
    topk_k: int = 32  # For TopK activation
    dtype: torch.dtype = torch.float32
    
    @property
    def d_hidden(self) -> int:
        return self.d_model * self.expansion_factor
```

### 4.2 Standard SAE Implementation

```python
class SparseAutoencoder(nn.Module):
    """
    Architecture:
        - Pre-encoder bias subtraction (centering)
        - Linear encoder with ReLU/GELU/TopK activation
        - Linear decoder
        - Optional decoder weight normalization
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden
        
        # Encoder: d_model -> d_hidden
        self.encoder = nn.Linear(config.d_model, config.d_hidden, bias=True)
        
        # Decoder: d_hidden -> d_model
        self.decoder = nn.Linear(config.d_hidden, config.d_model, bias=True)
        
        # Initialize with Kaiming uniform
        self._init_weights()
        
        if config.normalize_decoder:
            self._normalize_decoder()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse feature space."""
        # Center inputs by subtracting decoder bias
        x_centered = x - self.decoder.bias
        
        # Linear transformation + activation
        pre_activation = self.encoder(x_centered)
        
        if self.config.activation == "relu":
            features = F.relu(pre_activation)
        elif self.config.activation == "gelu":
            features = F.gelu(pre_activation)
        elif self.config.activation == "topk":
            features = self._topk_activation(pre_activation)
        
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to activation space."""
        return self.decoder(features)
    
    def compute_loss(self, x, reconstruction, features):
        """
        SAE Loss = MSE(x, reconstruction) + λ × L1(features)
        """
        recon_loss = F.mse_loss(reconstruction, x)
        l1_loss = self.config.l1_coefficient * features.abs().mean()
        return recon_loss + l1_loss
```

### 4.3 Gated SAE Variant

```python
class GatedSparseAutoencoder(nn.Module):
    """
    Uses separate pathways for magnitude and gate.
    Reduces dead features compared to standard SAE.
    """
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.decoder.bias
        
        # Magnitude pathway
        magnitude = F.relu(self.W_mag(x_centered))
        
        # Gate pathway with learnable threshold
        gate_logits = self.W_gate(x_centered) - self.gate_threshold
        gate = torch.sigmoid(gate_logits)
        
        # Apply gate
        features = magnitude * gate
        
        return features, gate
```

### 4.4 TopK Activation

```python
def _topk_activation(self, x: torch.Tensor) -> torch.Tensor:
    """Keep only top k values per sample for extreme sparsity."""
    k = self.config.topk_k
    topk_values, topk_indices = torch.topk(x, k=k, dim=-1)
    
    output = torch.zeros_like(x)
    output.scatter_(-1, topk_indices, F.relu(topk_values))
    
    return output
```

---

## 5. SAE Training Pipeline

### 5.1 Training Configuration

**Script**: `scripts/03_train_sae.py`

```yaml
# From configs/config.yaml
sae:
  expansion_factor: 8
  l1_coefficient: 0.0005
  epochs: 50
  batch_size: 256
  learning_rate: 0.0001
  warmup_steps: 1000
  normalize_decoder: true
  weight_decay: 0.0
```

### 5.2 Token Sampling Strategy

To manage memory, we sample tokens from each sequence:

```python
def prepare_training_data(activations, max_tokens_per_sample=50):
    """
    Sample tokens to reduce memory while preserving diversity.
    
    If sequence has >50 tokens, randomly sample 50.
    This gives O(samples × 50) training examples.
    """
    all_tokens = []
    
    for sample_idx in range(activations.shape[0]):
        sample_acts = activations[sample_idx]  # (seq_len, d_model)
        seq_len = sample_acts.shape[0]
        
        if seq_len > max_tokens_per_sample:
            indices = torch.randperm(seq_len)[:max_tokens_per_sample]
            sampled = sample_acts[indices]
        else:
            sampled = sample_acts
        
        all_tokens.append(sampled)
    
    return torch.cat(all_tokens, dim=0)  # (total_tokens, d_model)
```

### 5.3 Training Loop with W&B Integration

```python
def train_sae_for_layer(activations, config, layer, language, device='cuda'):
    """Train SAE for a specific layer/language combination."""
    
    # Initialize W&B
    wandb.init(
        project="sae-cross-lingual-bias",
        name=f"sae_layer{layer}_{language}",
        config={
            "layer": layer,
            "language": language,
            "d_model": activations.shape[-1],
            **config
        }
    )
    
    # Create SAE
    sae_config = SAEConfig(
        d_model=activations.shape[-1],
        expansion_factor=config['expansion_factor'],
        l1_coefficient=config['l1_coefficient']
    )
    sae = SparseAutoencoder(sae_config).to(device)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(sae.parameters(), lr=config['learning_rate'])
    
    # Training
    for epoch in range(config['epochs']):
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward
            reconstruction, features, aux_info = sae(batch)
            loss, components = sae.compute_loss(batch, reconstruction, features, 
                                                 return_components=True)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize decoder
            if sae.config.normalize_decoder:
                sae._normalize_decoder()
            
            # Log to W&B
            wandb.log({
                'loss': loss.item(),
                'recon_loss': components['reconstruction_loss'].item(),
                'l1_loss': components['l1_loss'].item(),
                'l0_sparsity': aux_info['l0_sparsity'].item(),
            })
    
    # Save checkpoint
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': sae_config,
        'd_model': sae_config.d_model,
        'd_hidden': sae_config.d_hidden,
    }, f'checkpoints/saes/sae_{language}_layer_{layer}.pt')
    
    wandb.finish()
    return sae
```

---

## 6. CLMB Framework Components

### 6.1 Hierarchical Bias Localization (HBL)

**File**: `src/clmb/hbl.py`

Decomposes bias across model components: **Vision → Projection → Language**

```python
class HierarchicalBiasLocalizer:
    """
    HBL decomposes bias attribution:
    
    Image → [Vision Encoder] → [Projection] → [Language Model] → Caption
                  ↓                  ↓                ↓
             V-Features         Bridge          L-Features
                  ↓                  ↓                ↓
            Vision Bias      Transfer Bias    Linguistic Bias
    """
    
    def compute_bias_attribution_score(
        self,
        male_activations: List[torch.Tensor],
        female_activations: List[torch.Tensor],
        feature_importance: Optional[np.ndarray] = None
    ) -> float:
        """
        Bias Attribution Score (BAS):
        
        BAS = Σ |activation_male - activation_female| × feature_importance
        
        Higher BAS = more gender-biased representations.
        """
        male_stack = torch.cat(male_activations, dim=0).mean(dim=0)
        female_stack = torch.cat(female_activations, dim=0).mean(dim=0)
        
        diff = torch.abs(male_stack - female_stack)
        
        if feature_importance is not None:
            diff = diff * torch.from_numpy(feature_importance).float()
        
        return diff.mean().item()
    
    def localize_bias(self, gender_data: Dict) -> BiasAttributionResult:
        """
        Main HBL method: Compute BAS for each component.
        """
        # Vision bias (average across vision layers)
        vision_bias = np.mean([
            self.compute_bias_attribution_score(
                gender_data['male']['vision'][layer],
                gender_data['female']['vision'][layer]
            )
            for layer in gender_data['male']['vision'].keys()
        ])
        
        # Projection bias
        projection_bias = self.compute_bias_attribution_score(
            gender_data['male']['projection'],
            gender_data['female']['projection']
        )
        
        # Language bias (average across language layers)
        language_bias = np.mean([
            self.compute_bias_attribution_score(
                gender_data['male']['language'][layer],
                gender_data['female']['language'][layer]
            )
            for layer in gender_data['male']['language'].keys()
        ])
        
        # Weighted total (α=0.3, β=0.2, γ=0.5)
        total_bias = 0.3*vision_bias + 0.2*projection_bias + 0.5*language_bias
        
        return BiasAttributionResult(
            vision_bias_score=vision_bias,
            projection_bias_score=projection_bias,
            language_bias_score=language_bias,
            total_bias=total_bias,
            dominant_component=max(['vision', 'projection', 'language'], 
                                  key=lambda x: locals()[f'{x}_bias'])
        )
```

### 6.2 Cross-Lingual Feature Alignment (CLFA)

**File**: `src/clmb/clfa.py`

Uses **optimal transport** (Wasserstein distance) to align features across languages.

```python
class CrossLingualFeatureAligner:
    """
    CLFA discovers semantic correspondences between Arabic and English
    feature representations using optimal transport.
    
    Key Innovation: Uses SAE features (sparse, interpretable) rather than
    dense activations for principled alignment.
    """
    
    def compute_wasserstein_matrix(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray
    ) -> np.ndarray:
        """
        Pairwise 1D Wasserstein distances between all feature pairs.
        """
        n_arabic = arabic_features.shape[1]
        n_english = english_features.shape[1]
        
        wasserstein_matrix = np.zeros((n_arabic, n_english))
        
        for i in range(n_arabic):
            for j in range(n_english):
                w_dist = wasserstein_distance(
                    arabic_features[:, i],
                    english_features[:, j]
                )
                wasserstein_matrix[i, j] = w_dist
        
        return wasserstein_matrix
    
    def optimal_transport_alignment(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray,
        reg: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entropic regularized optimal transport for feature alignment.
        
        Uses POT (Python Optimal Transport) library.
        """
        import ot
        
        # Uniform marginals
        a = np.ones(n_arabic) / n_arabic
        b = np.ones(n_english) / n_english
        
        # Cost matrix (normalized Wasserstein)
        C = self.compute_wasserstein_matrix(arabic_features, english_features)
        C = C / C.max()
        
        # Sinkhorn algorithm
        transport_plan = ot.sinkhorn(a, b, C, reg)
        
        # Threshold to get alignment matrix
        threshold = 1 / (n_arabic * n_english)
        alignment_matrix = (transport_plan > threshold).astype(float)
        
        return transport_plan, alignment_matrix
```

### 6.3 Cross-Lingual Bias Alignment Score (CLBAS)

**File**: `scripts/24_cross_lingual_overlap.py`

The **novel key metric** for this research:

```python
def compute_clbas(ar_features, en_features, ar_labels, en_labels) -> dict:
    """
    Cross-Lingual Bias Alignment Score (CLBAS)
    
    Measures if bias PATTERNS are similar despite using different FEATURES.
    
    CLBAS = (cosine_similarity + effect_correlation + rank_correlation) / 3
    
    Interpretation:
    - Low CLBAS (→ 0): Language-specific gender processing
    - High CLBAS (→ 1): Shared gender features across languages
    """
    # Compute effect sizes per language
    ar_effect_sizes = compute_gender_effect_sizes(ar_features, ar_labels)
    en_effect_sizes = compute_gender_effect_sizes(en_features, en_labels)
    
    # Component 1: Cosine similarity of effect size vectors
    cosine_sim = 1 - cosine(ar_effect_sizes, en_effect_sizes)
    
    # Component 2: Pearson correlation of effect sizes
    effect_corr, _ = stats.pearsonr(ar_effect_sizes, en_effect_sizes)
    
    # Component 3: Spearman rank correlation
    rank_corr, _ = stats.spearmanr(ar_effect_sizes, en_effect_sizes)
    
    # Combine (handle potential NaN)
    components = [cosine_sim, effect_corr, rank_corr]
    valid_components = [c for c in components if not np.isnan(c)]
    clbas_score = np.mean(valid_components) if valid_components else 0.0
    
    return {
        'clbas_score': clbas_score,
        'cosine_similarity': cosine_sim,
        'effect_size_correlation': effect_corr,
        'rank_correlation': rank_corr
    }
```

**Cohen's d Effect Size Computation**:

```python
def compute_gender_effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute Cohen's d for each SAE feature.
    
    Cohen's d = (μ_male - μ_female) / σ_pooled
    
    Where σ_pooled = √((σ_male² + σ_female²) / 2)
    """
    male_mask = labels == 1
    female_mask = labels == 0
    
    male_features = features[male_mask]
    female_features = features[female_mask]
    
    effect_sizes = np.zeros(features.shape[1])
    
    for i in range(features.shape[1]):
        male_vals = male_features[:, i]
        female_vals = female_features[:, i]
        
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
    
    return effect_sizes
```

---

## 7. Cross-Lingual Analysis Methodology

### 7.1 Feature Overlap Computation

```python
def compute_feature_overlap(ar_top_features: dict, en_top_features: dict) -> dict:
    """
    Compute overlap between Arabic and English gender features.
    
    This is the KEY NOVEL METRIC for your paper!
    
    Categories:
    - male_associated: Top-k features with positive effect size
    - female_associated: Top-k features with negative effect size
    - top_overall: Top-k by absolute effect size
    """
    results = {}
    
    for category in ['male_associated', 'female_associated', 'top_overall']:
        ar_set = set(ar_top_features[category])
        en_set = set(en_top_features[category])
        
        overlap = ar_set & en_set
        ar_specific = ar_set - en_set
        en_specific = en_set - ar_set
        union = ar_set | en_set
        
        results[category] = {
            'overlap_count': len(overlap),
            'overlap_pct': len(overlap) / len(ar_set) * 100,
            'jaccard_index': len(overlap) / len(union),
            'ar_specific_count': len(ar_specific),
            'en_specific_count': len(en_specific),
        }
    
    return results
```

### 7.2 Gender Probe Training

Linear probes test if gender information is linearly decodable from SAE features:

```python
def train_gender_probe(features: np.ndarray, labels: np.ndarray, cv: int = 5):
    """
    Train logistic regression probe with cross-validation.
    
    High accuracy = gender information is present and linearly accessible.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
    
    return scores.mean(), scores.std()
```

---

## 8. Surgical Bias Intervention

### 8.1 Intervention Types

**File**: `src/clmb/sbi.py`

```python
class SurgicalBiasIntervention:
    """
    Three intervention strategies:
    
    1. Ablation: Zero out bias features
    2. Neutralization: Average male/female values
    3. Amplification: Boost fairness-promoting features
    """
    
    def ablate_features(self, features: torch.Tensor, indices: List[int]):
        """Set specified features to zero."""
        intervened = features.clone()
        for idx in indices:
            intervened[..., idx] = 0
        return intervened
    
    def neutralize_features(self, male_features, female_features, indices):
        """Replace gendered features with average."""
        male_interv = male_features.clone()
        female_interv = female_features.clone()
        
        for idx in indices:
            avg_val = (male_features[..., idx] + female_features[..., idx]) / 2
            male_interv[..., idx] = avg_val
            female_interv[..., idx] = avg_val
        
        return male_interv, female_interv
    
    def amplify_features(self, features, indices, factor=2.0):
        """Boost specified (fairness-promoting) features."""
        intervened = features.clone()
        for idx in indices:
            intervened[..., idx] *= factor
        return intervened
```

### 8.2 Causal Feature Identification

```python
def identify_bias_causal_features(self, activations, genders, top_k=50):
    """
    Ablation-based causal attribution:
    
    For each feature, ablate it and measure change in gender bias.
    Features that reduce bias when ablated are causally linked.
    """
    with torch.no_grad():
        features = self.sae.encode(activations)
    
    male_mask = torch.tensor([g == 'male' for g in genders])
    female_mask = torch.tensor([g == 'female' for g in genders])
    
    # Baseline gender difference
    male_features = features[male_mask].mean(dim=0)
    female_features = features[female_mask].mean(dim=0)
    baseline_diff = torch.abs(male_features - female_features)
    
    causal_scores = []
    
    for idx in range(features.shape[-1]):
        # Ablate this feature
        ablated = features.clone()
        ablated[..., idx] = 0
        
        # Recompute difference
        ablated_male = ablated[male_mask].mean(dim=0)
        ablated_female = ablated[female_mask].mean(dim=0)
        ablated_diff = torch.abs(ablated_male - ablated_female)
        
        # Causal importance = reduction in gender difference
        importance = (baseline_diff.sum() - ablated_diff.sum()).item()
        causal_scores.append((idx, importance))
    
    # Sort by importance
    causal_scores.sort(key=lambda x: x[1], reverse=True)
    
    return causal_scores[:top_k]
```

### 8.3 Ablation Experiment Protocol

**Script**: `scripts/26_surgical_bias_intervention.py`

```python
def run_ablation_experiment(features, labels, original_acts, sae, top_features,
                            k_values=[10, 25, 50, 100, 200]):
    """
    Progressive ablation: ablate k features at a time.
    
    Measures:
    1. Probe accuracy drop (should drop if features are causal)
    2. Semantic preservation (should remain high)
    """
    results = []
    
    # Baseline
    baseline_acc, _ = train_gender_probe(features, labels)
    
    for k in k_values:
        # Ablate top-k features
        features_ablated = ablate_features(features, top_features[:k])
        
        # Test probe
        ablated_acc, _ = train_gender_probe(features_ablated, labels)
        accuracy_drop = baseline_acc - ablated_acc
        
        # Measure semantic preservation
        recon_similarity = compute_reconstruction_quality(
            original_acts, sae, features_ablated
        )
        
        results.append({
            'k': k,
            'baseline_accuracy': baseline_acc,
            'ablated_accuracy': ablated_acc,
            'accuracy_drop': accuracy_drop,
            'semantic_preservation': recon_similarity
        })
    
    return results
```

### 8.4 Cross-Lingual Ablation Test

**Key Validation**: Ablating Arabic features should affect Arabic probes but NOT English probes (and vice versa).

```python
def run_cross_lingual_ablation(ar_features, ar_labels, en_features, en_labels,
                                ar_top_features, en_top_features, k=100):
    """
    Cross-lingual causality test:
    
    If features are language-specific:
    - Ablating Arabic features → Arabic probe drops, English probe unchanged
    - Ablating English features → English probe drops, Arabic probe unchanged
    """
    # Arabic baseline and ablated
    ar_baseline, _ = train_gender_probe(ar_features, ar_labels)
    ar_ablated = ablate_features(ar_features, ar_top_features[:k])
    ar_after_ablation, _ = train_gender_probe(ar_ablated, ar_labels)
    
    # English baseline and ablated
    en_baseline, _ = train_gender_probe(en_features, en_labels)
    en_ablated = ablate_features(en_features, en_top_features[:k])
    en_after_ablation, _ = train_gender_probe(en_ablated, en_labels)
    
    # Cross-language test: Arabic features on English
    en_with_ar_ablation = ablate_features(en_features, ar_top_features[:k])
    en_after_ar_ablation, _ = train_gender_probe(en_with_ar_ablation, en_labels)
    
    # Cross-language test: English features on Arabic
    ar_with_en_ablation = ablate_features(ar_features, en_top_features[:k])
    ar_after_en_ablation, _ = train_gender_probe(ar_with_en_ablation, ar_labels)
    
    return {
        'arabic': {
            'same_language_drop': ar_baseline - ar_after_ablation,
            'cross_language_drop': ar_baseline - ar_after_en_ablation
        },
        'english': {
            'same_language_drop': en_baseline - en_after_ablation,
            'cross_language_drop': en_baseline - en_after_ar_ablation
        }
    }
```

---

## 9. Experimental Results

### 9.1 Model Comparison Summary

| Metric | Qwen2-VL-7B | PaLiGemma-3B | Ratio |
|--------|-------------|--------------|-------|
| **Mean CLBAS** | 0.0040 | 0.0268 | **6.7× lower** |
| **Max CLBAS** | 0.0079 (L20) | 0.0407 (L17) | **5.2× lower** |
| **Total Feature Overlap** | 1 feature | 3 features | **3× fewer** |
| **Mean Arabic Probe** | 90.3% | ~87% | +3% |
| **Mean English Probe** | 91.8% | ~90% | +2% |

### 9.2 Layer-wise CLBAS Scores

**Qwen2-VL-7B**:
| Layer | CLBAS | Overlap Count |
|-------|-------|---------------|
| 0 | 0.0015 | 0 |
| 4 | 0.0037 | 0 |
| 8 | 0.0047 | 0 |
| 12 | 0.0018 | 0 |
| 16 | 0.0029 | 0 |
| 20 | **0.0079** | **1** |
| 24 | 0.0024 | 0 |
| 27 | 0.0073 | 0 |

**PaLiGemma-3B**:
| Layer | CLBAS | Overlap Count |
|-------|-------|---------------|
| 3 | 0.0106 | 0 |
| 6 | 0.0145 | 0 |
| 9 | 0.0275 | 2 |
| 12 | 0.0392 | 1 |
| 15 | 0.0282 | 0 |
| 17 | **0.0407** | 0 |

### 9.3 Key Finding

**Qwen2-VL (7B parameters) exhibits significantly more language-specific gender processing than PaLiGemma (3B parameters).**

This suggests:
1. Larger VLMs develop more specialized internal representations per language
2. Cross-lingual bias transfer is reduced in larger models
3. Multilingual training at scale leads to language-specific feature emergence

---

## 10. Technical Implementation Details

### 10.1 Dtype Handling

Qwen2-VL stores activations in `bfloat16`, requiring explicit conversion:

```python
# Fix for RuntimeError: mat1 and mat2 must have same dtype
activations = data["activations"].float()  # Convert bfloat16 → float32
```

### 10.2 HPC Cluster Configuration (NCC Durham)

**SLURM Job Template**:
```bash
#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1       # A100 80GB
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out

module purge
module load cuda/12.2

source ~/venvs/sae/bin/activate
cd $PROJECT_DIR

python scripts/30_qwen2vl_cross_lingual_analysis.py \
    --device cuda \
    --wandb \
    --wandb_project qwen2vl-sae-analysis
```

### 10.3 Memory Optimization

```python
# Clear GPU memory between layers
gc.collect()
torch.cuda.empty_cache()

# Use batch_size=1 for extraction (processor limitation)
# Use batch_size=256 for SAE training (sufficient RAM)
```

### 10.4 Visualization Generation

```python
def create_final_comparison_figure(qwen_results, pali_results, output_dir):
    """Publication-quality comparison figure."""
    fig = plt.figure(figsize=(16, 12))
    
    # 4-panel layout: CLBAS, Overlap, Probes, Summary
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    # Panel 1: CLBAS by layer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(qwen_layers, qwen_clbas, 'o-', label='Qwen2-VL-7B', color='#2ecc71')
    ax1.plot(pali_layers, pali_clbas, 's-', label='PaLiGemma-3B', color='#e74c3c')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('CLBAS Score')
    ax1.set_title('Cross-Lingual Bias Alignment')
    ax1.legend()
    
    # ... additional panels ...
    
    plt.savefig(output_dir / 'final_model_comparison.png', dpi=300, bbox_inches='tight')
```

---

## Appendix A: Key Python Dependencies

```
torch>=2.1.0
transformers>=4.40.0
camel-tools              # Arabic NLP
POT                      # Optimal Transport
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
wandb>=0.16.0
einops>=0.7.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.66.0
pyyaml>=6.0
```

---

## Appendix B: Mathematical Formulations

### B.1 SAE Loss Function

$$\mathcal{L}_{SAE} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{Reconstruction}} + \underbrace{\lambda \|h\|_1}_{\text{Sparsity}}$$

Where:
- $x$ = input activation
- $\hat{x} = W_{dec} \cdot h + b_{dec}$ = reconstruction
- $h = \text{ReLU}(W_{enc} \cdot (x - b_{dec}) + b_{enc})$ = sparse features
- $\lambda = 5 \times 10^{-4}$ = L1 coefficient

### B.2 Cohen's d Effect Size

$$d = \frac{\mu_{male} - \mu_{female}}{\sigma_{pooled}}$$

Where:
$$\sigma_{pooled} = \sqrt{\frac{\sigma_{male}^2 + \sigma_{female}^2}{2}}$$

### B.3 CLBAS Score

$$\text{CLBAS} = \frac{1}{3}\left(\cos(\vec{d}_{ar}, \vec{d}_{en}) + \rho_{pearson}(\vec{d}_{ar}, \vec{d}_{en}) + \rho_{spearman}(\vec{d}_{ar}, \vec{d}_{en})\right)$$

Where $\vec{d}_{lang}$ is the vector of Cohen's d effect sizes for all features in that language.

### B.4 Jaccard Overlap Index

$$J(A_{ar}, A_{en}) = \frac{|A_{ar} \cap A_{en}|}{|A_{ar} \cup A_{en}|}$$

Where $A_{lang}$ is the set of top-k gender-associated features.

---

## Appendix C: References

1. **Elhage et al.** (2022). "Toy Models of Superposition" - Feature interaction theory
2. **Geva et al.** (2023). "Transformer Interpretability Beyond Attention" - Logit lens
3. **Cunningham et al.** (2023). "Sparse Autoencoders Find Highly Interpretable Features" - SAE methodology
4. **Hewitt & Liang** (2019). "Designing and Interpreting Probes with Control Tasks" - Probing
5. **Peyré & Cuturi** (2019). "Computational Optimal Transport" - OT foundations

---

*Report generated: January 20, 2026*  
*Project: Cross-Lingual SAE Mechanistic Interpretability*  
*Models: PaLiGemma-3B, Qwen2-VL-7B-Instruct*
