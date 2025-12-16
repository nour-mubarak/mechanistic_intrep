# Weights & Biases Tracking

## Project Information

**Project**: `sae-captioning-bias`
**Entity**: `nourmubarak`
**Current Run**: `comprehensive-analysis-20251216-125637`

🚀 **View Live Dashboard**: https://wandb.ai/nourmubarak/sae-captioning-bias/runs/9luh3v40

## Tracked Metrics

### Per-Layer Metrics

For each layer (10, 14, 18, 22), we track:

#### Feature Statistics
- `layer_{N}/english_l0` - Average number of active features per sample (English)
- `layer_{N}/arabic_l0` - Average number of active features per sample (Arabic)
- `layer_{N}/english_dead_features` - Number of never-activated features (English)
- `layer_{N}/arabic_dead_features` - Number of never-activated features (Arabic)
- `layer_{N}/english_mean_activation` - Mean activation across all features (English)
- `layer_{N}/arabic_mean_activation` - Mean activation across all features (Arabic)

#### Prisma Mechanistic Analysis
- `layer_{N}/english_rank` - Effective rank of English feature matrix
- `layer_{N}/arabic_rank` - Effective rank of Arabic feature matrix
- `layer_{N}/rank_difference` - Absolute difference in effective ranks
- `layer_{N}/english_info_content` - Information content (entropy-based)
- `layer_{N}/arabic_info_content` - Information content (entropy-based)

#### Cross-Lingual Alignment
- `layer_{N}/aligned_features` - Number of aligned feature pairs
- `layer_{N}/alignment_ratio` - Proportion of features that align cross-lingually
- `layer_{N}/mean_similarity` - Average cosine similarity of aligned features

#### Gender Bias Correlation
- `layer_{N}/gender_bias_correlation` - Correlation of gender bias between languages

### Visualizations Logged

For each layer:

1. **Statistics Plot** (`layer_{N}/statistics_plot`)
   - 9-panel comprehensive feature analysis
   - Activation frequencies, distributions, correlations
   - Dead features, sparsity metrics
   - Gender-specific patterns

2. **Gender Bias Plot** (`layer_{N}/gender_bias_plot`)
   - Top male-biased features (English & Arabic)
   - Top female-biased features (English & Arabic)
   - Effect sizes (Cohen's d)

3. **Embeddings Plot** (`layer_{N}/embeddings_plot`)
   - PCA projection of feature space
   - t-SNE projection of feature space
   - Language separation visualization

### Summary Table

At the end of the run, a summary table is logged with:
- Layer number
- English & Arabic L0 (sparsity)
- English & Arabic dead features count
- Feature alignment percentage
- Gender bias correlation

## Key Research Questions Tracked

### 1. Feature Sparsity
- **Metric**: L0 per sample
- **Interpretation**: Lower L0 = more sparse/interpretable representations
- **Expected**: ~32 active features (due to Top-K constraint)

### 2. Dead Features
- **Metric**: Count of features with zero activation
- **Interpretation**: Indicates dictionary utilization efficiency
- **Ideal**: Minimal dead features (~0-5%)

### 3. Cross-Lingual Alignment
- **Metric**: Alignment ratio
- **Interpretation**:
  - High (>70%): Universal gender representation
  - Low (<30%): Language-specific encoding
  - Medium: Mixed representations

### 4. Gender Bias Patterns
- **Metric**: Cross-lingual correlation of bias
- **Interpretation**:
  - High positive correlation: Consistent bias across languages
  - Low/negative correlation: Different bias patterns

### 5. Effective Rank
- **Metric**: Rank at 95% variance threshold
- **Interpretation**:
  - Lower rank: More structured/redundant representations
  - Higher rank: More distributed representations
  - Similar ranks: Comparable complexity

## Expected Patterns

### Hypothesis 1: Universal Gender Encoding
If true, expect:
- High alignment ratio (>60%)
- High gender bias correlation (>0.7)
- Similar effective ranks

### Hypothesis 2: Language-Specific Encoding
If true, expect:
- Low alignment ratio (<40%)
- Low gender bias correlation (<0.3)
- Different effective ranks

### Hypothesis 3: Layer-Wise Evolution
Expect changes across layers:
- Early layers (10): More language-specific
- Middle layers (14, 18): Mixed
- Late layers (22): More universal/semantic

## Dashboard Organization

### Recommended Panels

1. **Sparsity Overview**
   - Line plot: L0 across layers (English vs Arabic)
   - Bar chart: Dead features per layer

2. **Alignment Analysis**
   - Line plot: Alignment ratio across layers
   - Scatter plot: English rank vs Arabic rank

3. **Gender Bias**
   - Line plot: Gender correlation across layers
   - Heatmap: Gender bias patterns

4. **Visualizations Gallery**
   - Grid of all statistics plots
   - Grid of all embedding plots

## Tags

- `comprehensive`: Full analysis pipeline
- `prisma`: ViT-Prisma mechanistic analysis included
- `gender-bias`: Gender bias focus
- `cross-lingual`: English-Arabic comparison

## Configuration Logged

```yaml
model: google/gemma-3-4b-it
layers: [10, 14, 18, 22]
num_samples: 500
sae_expansion: 8x (2560 → 20480)
sae_topk: 32
```

## Real-Time Monitoring

While the analysis runs, you can:

1. **Monitor Progress**: Watch metrics appear in real-time
2. **Compare Layers**: See how patterns evolve
3. **Download Results**: Export all metrics and plots
4. **Share Results**: Generate report links for collaborators

---

**Analysis Status**: 🔄 Running
**Current Layer**: 10/22
**Est. Completion**: ~30-45 minutes (including t-SNE computation)
