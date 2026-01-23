# Cosine Similarity: Literature Support and Defense

## Summary

**Cosine Similarity** is the standard metric for measuring cross-lingual alignment in neural network representations. It is well-established with extensive literature support.

---

## Key References (For Citation)

### 1. Conneau et al. (2020) - ACL Main Conference
**"Emerging Cross-lingual Structure in Pretrained Language Models"**
- **Venue**: ACL 2020 (333 citations)
- **Usage**: Uses cosine similarity for cross-lingual alignment in mBERT
- **Quote**: "We use nearest neighbor search and cosine similarity to measure cross-lingual similarity"
- **Link**: https://aclanthology.org/2020.acl-main.536/

### 2. Hämmerl et al. (2024) - ACL Findings
**"Understanding Cross-Lingual Alignment -- A Survey"**
- **Venue**: ACL Findings 2024 (36 citations)
- **Usage**: Comprehensive survey recommending cosine similarity
- **Quote**: "Cross-lingual alignment means that words with similar semantics are more similar in the representation... Using cosine similarity"
- **Link**: https://arxiv.org/abs/2404.06228

### 3. Wang et al. (2018) - EMNLP
**"Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks"**
- **Venue**: EMNLP 2018 (806 citations)
- **Usage**: Uses cosine similarity for cross-lingual entity alignment
- **Link**: https://aclanthology.org/D18-1032/

### 4. Ulčar & Robnik-Šikonja (2022) - Neural Computing and Applications
**"Cross-lingual alignments of ELMo contextual embeddings"**
- **Venue**: Neural Computing and Applications (22 citations)
- **Usage**: "We measured the cosine distance between vectors"
- **Link**: https://link.springer.com/article/10.1007/s00521-022-07164-x

### 5. Pallucchini et al. (2025) - ACM Computing Surveys
**"Lost in Alignment: A Survey on Cross-lingual Alignment Methods"**
- **Venue**: ACM Computing Surveys (2 citations - very recent)
- **Usage**: Reviews alignment methods including cosine similarity
- **Link**: https://dl.acm.org/doi/abs/10.1145/3764112

---

## Formula

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- **A** = Arabic gender effect size vector (Cohen's d for each SAE feature)
- **B** = English gender effect size vector (Cohen's d for each SAE feature)

---

## Why Cosine Similarity?

### 1. **Scale-Invariant**
- Measures direction (angle), not magnitude
- Effect sizes can vary in scale across languages; cosine handles this

### 2. **Bounded [-1, 1]**
- Easy to interpret:
  - **1.0** = Perfect alignment (identical directions)
  - **0.0** = Orthogonal (no alignment)
  - **-1.0** = Opposite directions

### 3. **Standard in NLP**
- Used in BERT, mBERT, XLM-R cross-lingual studies
- Default metric in Hugging Face similarity functions

### 4. **Widely Cited**
- 1000+ combined citations across the key papers
- Accepted at top venues (ACL, EMNLP, NAACL)

---

## How to Defend in Paper/Presentation

### In Methods Section:
> "We measure cross-lingual alignment using cosine similarity between gender effect size vectors, following established methodology for cross-lingual representation analysis (Conneau et al., 2020; Hämmerl et al., 2024)."

### If Asked "Why Cosine Similarity?":
> "Cosine similarity is the standard metric for cross-lingual alignment in neural networks, used in foundational work on multilingual BERT (Conneau et al., 2020, ACL) and recommended in recent surveys (Hämmerl et al., 2024, ACL Findings). It is scale-invariant, bounded, and interpretable - ideal for comparing feature directions across languages with potentially different effect magnitudes."

### If Asked "Why Not Other Metrics?":
> "We considered alternatives like CKA (Centered Kernel Alignment) and RSA (Representational Similarity Analysis), but cosine similarity is more interpretable for our vector-to-vector comparison and is the standard in cross-lingual NLP literature. Our results are robust - we also report Jaccard index for feature set overlap, which yields consistent conclusions."

---

## Our Results

| Model | Mean Cosine Sim | Interpretation |
|-------|-----------------|----------------|
| PaLiGemma-3B | 0.027 | Near-zero alignment |
| Qwen2-VL-7B | 0.004 | Extremely low alignment |
| **Ratio** | **6.7×** | Larger model = more specific |

**Key Finding**: Near-zero cosine similarity (0.004-0.027) indicates that Arabic and English use **language-specific feature circuits** for gender encoding, not shared representations.

---

## BibTeX Citations

```bibtex
@inproceedings{conneau-etal-2020-emerging,
    title = "Emerging Cross-lingual Structure in Pretrained Language Models",
    author = "Conneau, Alexis and Wu, Shijie and Li, Haoran and Zettlemoyer, Luke and Stoyanov, Veselin",
    booktitle = "Proceedings of the 58th Annual Meeting of the ACL",
    year = "2020",
    pages = "6022--6034"
}

@inproceedings{hammerl-etal-2024-understanding,
    title = "Understanding Cross-Lingual Alignment -- A Survey",
    author = "Hämmerl, Katharina and Libovický, Jindřich and Fraser, Alexander",
    booktitle = "Findings of the ACL",
    year = "2024"
}

@inproceedings{wang-etal-2018-cross,
    title = "Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks",
    author = "Wang, Zhichun and Lv, Qingsong and Lan, Xiaohan and Zhang, Yu",
    booktitle = "Proceedings of EMNLP",
    year = "2018",
    pages = "349--357"
}
```

---

*Last updated: January 21, 2026*
