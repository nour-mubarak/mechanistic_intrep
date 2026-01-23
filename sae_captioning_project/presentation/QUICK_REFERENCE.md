# Quick Reference Card: Key Results Summary

## 📊 Main Numbers to Remember

### Cosine Similarity Scores (Lower = More Language-Specific)
| Model | Cosine Sim | Interpretation |
|-------|------------|----------------|
| **PaLiGemma-3B** | 0.027 | Very low alignment |
| **Qwen2-VL-7B** | 0.004 | Extremely low alignment |
| **Ratio** | 6.7× | Larger model = more specific |

### Feature Overlap
| Model | Total Overlap | % of Features |
|-------|---------------|---------------|
| **PaLiGemma-3B** | 3 features | 0.4% |
| **Qwen2-VL-7B** | 1 feature | 0.003% |

### Probe Accuracy
| Language | PaLiGemma | Qwen2-VL |
|----------|-----------|----------|
| **Arabic** | 86.5% | 90.3% |
| **English** | 93.0% | 91.8% |

### SBI Results (k=200 ablations)
- **Accuracy Drop**: 0% (within noise)
- **Implication**: Gender info is distributed

---

## 🎯 Three Key Findings

1. **Near-Zero Alignment** (Cosine Sim=0.027)
   - Gender features are language-specific
   - Arabic ≠ English processing

2. **Scaling Effect** (6.7× ratio)
   - Larger models → more specific
   - 7B has more distinct circuits

3. **Distributed Encoding** (0% drop)
   - No single "gender neuron"
   - Can't simply ablate features

---

## 📈 For Discussion

### Potential Questions:
1. Why does scaling increase specificity?
2. Cosine similarity is standard (Conneau et al. 2020)
3. Which languages next?
4. Target venue: ACL, EMNLP, NeurIPS?

### Limitations to Acknowledge:
- Binary gender only
- Two languages only
- No causal validation yet

---

## 📁 Key Files

```
presentation/
├── SUPERVISOR_PRESENTATION.md    # Full presentation
├── key_findings.png              # Summary figure
├── main_comparison.png           # Model comparison
├── methodology.png               # Pipeline overview
├── conclusions.png               # Final conclusions
├── publication_summary.png       # Complete results
└── sbi_accuracy_vs_k.png        # Ablation effects
```

---

**Project Location**: 
`/home2/jmsk62/mechanistic_intrep/mech_intrep/.../sae_captioning_project/`

**GitHub**: 
`https://github.com/nour-mubarak/mechanistic_intrep`

---

*Last updated: January 21, 2026*
