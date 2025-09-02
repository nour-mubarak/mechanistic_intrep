# Deployment Instructions

This document provides instructions for deploying and using the Enhanced Mechanistic Interpretability System.

## System Requirements

### Hardware Requirements
- **GPU:** NVIDIA GPU with at least 8GB VRAM (recommended: 16GB+ for larger models)
- **RAM:** Minimum 16GB system RAM (recommended: 32GB+)
- **Storage:** At least 10GB free disk space for models and results

### Software Requirements
- **Python:** 3.8 or higher
- **CUDA:** 11.0 or higher (for GPU acceleration)
- **Operating System:** Linux (Ubuntu 18.04+), macOS, or Windows 10+

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/enhanced-mechanistic-interpretability.git
cd enhanced-mechanistic-interpretability
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration

### 1. WandB Setup (Optional)
If you want to use Weights & Biases for experiment tracking:
```bash
wandb login
```

### 2. Configure Experiment
Edit `configs/experiment_config.yaml` to customize your experiment settings:
```yaml
total_samples: 1000
base_model_name: "Salesforce/blip-image-captioning-base"
use_wandb: true
project_name: "your-project-name"
```

## Running the System

### Quick Start
Run the complete experiment with default settings:
```bash
python -m enhanced_mechanistic_system.real_experiment
```

### Custom Configuration
1. Modify `real_experiment.py` to adjust the `ExperimentConfig`
2. Run the experiment:
```bash
python real_experiment.py
```

## Output Structure

After running the experiment, you'll find results in the `gender_bias_experiment_results/` directory:

```
gender_bias_experiment_results/
├── comprehensive_dashboard.html    # Interactive results dashboard
├── executive_summary.md           # High-level summary
├── baseline_results.json          # Baseline model evaluation
├── mechanistic_analysis.json      # Circuit discovery results
├── intervention_results.json      # Intervention experiment results
├── final_evaluation/              # Detailed evaluation reports
├── training_progress.png          # Training visualization
└── gender_distribution.png        # Dataset analysis
```

## Performance Optimization

### For Limited GPU Memory
- Reduce `batch_size` in `ExperimentConfig`
- Increase `gradient_accumulation_steps`
- Use smaller models (e.g., "Salesforce/blip-image-captioning-base")

### For Faster Training
- Increase `batch_size` if you have sufficient GPU memory
- Use multiple GPUs with `torch.nn.DataParallel`
- Enable mixed precision training

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory:** Reduce batch size or use CPU-only mode
2. **Import Errors:** Ensure all dependencies are installed correctly
3. **WandB Issues:** Check internet connection and API key

### Getting Help
- Check the `TROUBLESHOOTING.md` file for detailed solutions
- Review the API documentation in `API_DOCUMENTATION.md`
- Open an issue on the GitHub repository

## Production Deployment

### Docker Deployment
A Dockerfile is provided for containerized deployment:
```bash
docker build -t mechanistic-interpretability .
docker run --gpus all -v $(pwd)/results:/app/results mechanistic-interpretability
```

### Cloud Deployment
The system can be deployed on cloud platforms like:
- **Google Colab:** Use the provided notebook
- **AWS SageMaker:** Follow the SageMaker deployment guide
- **Azure ML:** Use the Azure ML configuration files

## Monitoring and Maintenance

### Experiment Tracking
- Use WandB dashboards to monitor training progress
- Set up alerts for experiment completion or failures
- Regularly backup experiment results

### Model Updates
- Periodically update the base models to newer versions
- Re-run experiments to compare performance improvements
- Update bias detection methods as new techniques become available

## Security Considerations

### Data Privacy
- Ensure image datasets comply with privacy regulations
- Use anonymized or synthetic data when possible
- Implement proper access controls for sensitive datasets

### Model Security
- Validate model outputs before deployment
- Implement bias monitoring in production systems
- Regular audits of model fairness metrics

## Support and Maintenance

### Regular Updates
- Keep dependencies updated for security patches
- Monitor for new bias detection techniques
- Update documentation as the system evolves

### Community Contributions
- Follow the contribution guidelines in `CONTRIBUTING.md`
- Submit bug reports and feature requests via GitHub issues
- Participate in community discussions and improvements

