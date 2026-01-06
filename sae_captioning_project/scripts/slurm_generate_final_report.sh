#!/bin/bash
#SBATCH --job-name=final_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/step_final_report_%j.out
#SBATCH --error=logs/step_final_report_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Final Step: Generate Comprehensive Report
# ==========================================
# Consolidates all analysis results into final report

echo "=========================================="
echo "Final Step: Generate Comprehensive Report"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
source venv/bin/activate

# Create final report directory
mkdir -p visualizations/FINAL_REPORT

# Generate comprehensive summary report
python << 'EOF'
import json
import os
from pathlib import Path
from datetime import datetime

print("Generating final comprehensive report...")

report_dir = Path("visualizations/FINAL_REPORT")
report_file = report_dir / f"COMPLETE_ANALYSIS_REPORT_{datetime.now().strftime('%Y%m%d')}.md"

# Collect all analysis results
analyses_completed = []

# Check each analysis step
analysis_dirs = [
    ("Comprehensive Analysis (All Layers)", "visualizations/comprehensive_all_layers"),
    ("Feature Interpretation", "visualizations/feature_interpretation"),
    ("Feature Ablation", "visualizations/feature_ablation"),
    ("Visual Patterns", "visualizations/visual_patterns_layer_10"),
    ("Feature Amplification", "visualizations/feature_amplification_layer_10"),
    ("Cross-Layer Analysis (Full)", "visualizations/cross_layer_analysis_full"),
    ("Cross-Layer Analysis (Key)", "visualizations/cross_layer_analysis_key"),
    ("Qualitative Analysis", "visualizations/qualitative_layer_10"),
]

with open(report_file, 'w') as f:
    f.write("# Complete Mechanistic Interpretability Analysis Report\n\n")
    f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model**: google/gemma-3-4b-it (34 layers)\n")
    f.write(f"**Dataset**: Full dataset (2000 samples)\n")
    f.write(f"**Methodology**: Durham NCC cluster with full layer extraction\n\n")
    f.write("---\n\n")

    f.write("## Pipeline Completion Status\n\n")

    for name, path in analysis_dirs:
        if Path(path).exists():
            file_count = len(list(Path(path).rglob('*.*')))
            analyses_completed.append(name)
            f.write(f"- ✅ **{name}**: {file_count} files generated\n")
        else:
            f.write(f"- ❌ **{name}**: Not found\n")

    f.write(f"\n**Total analyses completed**: {len(analyses_completed)}/{len(analysis_dirs)}\n\n")
    f.write("---\n\n")

    # Load and summarize comprehensive analysis results
    comp_results_file = Path("visualizations/comprehensive_all_layers/comprehensive_analysis_results.json")
    if comp_results_file.exists():
        f.write("## Key Findings Summary\n\n")
        f.write("### Gender-Biased Features Across All Layers\n\n")

        with open(comp_results_file) as rf:
            results = json.load(rf)

        f.write("Top gender-biased features identified per layer:\n\n")
        f.write("| Layer | Male-Biased Features | Female-Biased Features |\n")
        f.write("|-------|---------------------|------------------------|\n")

        for layer_str, layer_data in sorted(results.get('layers', {}).items(), key=lambda x: int(x[0])):
            male_features = layer_data.get('male_biased_features', [])[:3]
            female_features = layer_data.get('female_biased_features', [])[:3]

            male_str = ", ".join([str(f['feature']) for f in male_features])
            female_str = ", ".join([str(f['feature']) for f in female_features])

            f.write(f"| {layer_str} | {male_str} | {female_str} |\n")

        f.write("\n")

    # Load cross-layer analysis
    cross_layer_file = Path("visualizations/cross_layer_analysis_full/cross_layer_results.json")
    if cross_layer_file.exists():
        f.write("### Cross-Layer Evolution\n\n")

        with open(cross_layer_file) as rf:
            cross_results = json.load(rf)

        f.write("Gender bias evolution through network depth:\n\n")
        f.write("| Layer | Male Differential | Female Differential |\n")
        f.write("|-------|-------------------|---------------------|\n")

        for layer_str, data in sorted(cross_results.get('male_biased_ablation', {}).items(), key=lambda x: int(x[0])):
            male_diff = data.get('differential', 0)
            female_data = cross_results.get('female_biased_ablation', {}).get(layer_str, {})
            female_diff = female_data.get('differential', 0)

            f.write(f"| {layer_str} | {male_diff:.2f} | {female_diff:.2f} |\n")

        f.write("\n")

    f.write("---\n\n")
    f.write("## Visualization Outputs\n\n")
    f.write("All generated visualizations are available in:\n\n")

    for name, path in analysis_dirs:
        if Path(path).exists():
            f.write(f"### {name}\n")
            f.write(f"Location: `{path}/`\n\n")

            # List key files
            png_files = list(Path(path).glob('*.png'))[:5]
            if png_files:
                f.write("Key visualizations:\n")
                for png in png_files:
                    f.write(f"- `{png.name}`\n")
            f.write("\n")

    f.write("---\n\n")
    f.write("## Reproducibility\n\n")
    f.write("This analysis was conducted using the complete pipeline:\n\n")
    f.write("```bash\n")
    f.write("# Full pipeline execution\n")
    f.write("bash scripts/slurm_00_full_pipeline.sh\n")
    f.write("```\n\n")

    f.write("Individual steps:\n")
    f.write("1. Data preparation: `slurm_01_prepare_data.sh`\n")
    f.write("2. Activation extraction: `slurm_02_parallel_extraction.sh` (34 layers)\n")
    f.write("3. SAE training: `slurm_03_train_all_saes.sh` (34 SAEs)\n")
    f.write("4. Comprehensive analysis: `slurm_09_comprehensive_analysis.sh`\n")
    f.write("5-10. Advanced analyses: Steps 11-17\n\n")

    f.write("---\n\n")
    f.write("## Next Steps\n\n")
    f.write("- Review individual analysis reports in each output directory\n")
    f.write("- Examine visualizations for detailed insights\n")
    f.write("- Prepare manuscript with key findings\n")
    f.write("- Design debiasing interventions based on identified features\n\n")

print(f"Final report generated: {report_file}")

# Create summary statistics
print("\n" + "="*50)
print("PIPELINE COMPLETION SUMMARY")
print("="*50)
print(f"Analyses completed: {len(analyses_completed)}/{len(analysis_dirs)}")
print(f"Report location: {report_file}")
print("="*50)

EOF

echo ""
echo "=========================================="
echo "Report Generation Complete"
echo "=========================================="
echo ""
echo "Final report location:"
echo "  visualizations/FINAL_REPORT/"
echo ""
echo "All analyses complete!"
echo "Completed at: $(date)"
echo "=========================================="
