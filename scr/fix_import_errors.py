#!/usr/bin/env python3
"""
Quick fix script for common import and runtime errors in the mechanistic interpretability system.
Run this script to automatically apply fixes to your code.
"""

import os
import sys
from pathlib import Path

def fix_wandb_import(file_path):
    """Fix the WandBLogger import issue in real_experiment.py"""
    print(f"Fixing WandB import in {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the import
    content = content.replace(
        'from wandb_integration import WandBLogger',
        'from wandb_integration import WandbTracker'
    )
    
    # Fix the class instantiation
    content = content.replace(
        'self.wandb_logger = WandBLogger(',
        'self.wandb_logger = WandbTracker('
    )
    
    # Fix the initialization call
    content = content.replace(
        'experiment_name="gender_bias_mechanistic_analysis",\n                config=self.config.__dict__\n            )\n            self.wandb_logger.init_experiment()',
        'tags=["gender_bias_mechanistic_analysis"]\n            )\n            self.wandb_logger.init_experiment(self.config.__dict__)'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✓ WandB import fixed")

def fix_model_call(file_path):
    """Fix the model call issue in mechanistic analysis"""
    print(f"Fixing model call in {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the model call that was causing the error
    old_code = """            # Extract attention patterns
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, output_attentions=True)"""
    
    new_code = """            # First generate a caption to get input_ids
            with torch.no_grad():
                # Generate caption
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    do_sample=False
                )
                
                # Now run forward pass with both pixel_values and input_ids for attention analysis
                outputs = self.model(
                    pixel_values=pixel_values, 
                    input_ids=generated_ids,
                    output_attentions=True
                )"""
    
    content = content.replace(old_code, new_code)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✓ Model call fixed")

def fix_font_warnings(file_path):
    """Fix font warnings in visualizations.py"""
    print(f"Fixing font configuration in {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix font configuration
    content = content.replace(
        "plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']",
        "plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✓ Font configuration fixed")

def add_missing_method(file_path):
    """Add the missing log_experiment_results method to wandb_integration.py"""
    print(f"Adding missing method to {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if method already exists
    if 'def log_experiment_results' in content:
        print("✓ Method already exists")
        return
    
    # Add the missing method at the end of the class
    method_code = '''
    def log_experiment_results(self, results: Dict[str, Any]) -> None:
        """
        Log complete experiment results to wandb.
        
        Args:
            results: Complete experiment results dictionary
        """
        # Log final metrics
        if 'final_evaluation' in results:
            final_eval = results['final_evaluation']
            if 'final_results' in final_eval:
                final_results = final_eval['final_results']
                
                # Log bias metrics
                if hasattr(final_results, 'bias_metrics'):
                    bias_metrics = final_results.bias_metrics
                    wandb.log({
                        "final_gender_gap": bias_metrics.gender_gap,
                        "final_male_accuracy": bias_metrics.male_accuracy,
                        "final_female_accuracy": bias_metrics.female_accuracy,
                        "final_demographic_parity": bias_metrics.demographic_parity,
                        "final_equalized_odds": bias_metrics.equalized_odds
                    })
                
                # Log quality metrics
                if hasattr(final_results, 'quality_metrics'):
                    quality_metrics = final_results.quality_metrics
                    wandb.log({
                        "final_bleu_4": quality_metrics.bleu_4,
                        "final_rouge_l": quality_metrics.rouge_l,
                        "final_meteor": quality_metrics.meteor,
                        "final_bertscore_f1": quality_metrics.bertscore_f1
                    })
        
        # Log results as artifact
        self.log_results_artifact(results, "complete_experiment_results")
        
        print("Complete experiment results logged to wandb")
'''
    
    # Find the end of the WandbTracker class and add the method
    lines = content.split('\n')
    class_end_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith('def create_sample_experiment'):
            class_end_idx = i
            break
    
    if class_end_idx > 0:
        lines.insert(class_end_idx, method_code)
        content = '\n'.join(lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✓ Missing method added")
    else:
        print("⚠ Could not find insertion point for method")

def main():
    """Apply all fixes"""
    print("=" * 60)
    print("MECHANISTIC INTERPRETABILITY SYSTEM - ERROR FIXES")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path.cwd()
    
    # Check if we're in the right directory
    if not (current_dir / 'scr/real_experiment.py').exists():
        print("❌ Error: scr/real_experiment.py not found in current directory")
        print("Please run this script from the directory containing your project files")
        return 1
    
    try:
        # Apply fixes
        fix_wandb_import(current_dir / 'scr/real_experiment.py')
        fix_model_call(current_dir / 'scr/real_experiment.py')
        
        if (current_dir / 'scr/visualizations.py').exists():
            fix_font_warnings(current_dir / 'scr/visualizations.py')
        
        if (current_dir / 'scr/wandb_integration.py').exists():
            add_missing_method(current_dir / 'scr/wandb_integration.py')
        
        print("\n" + "=" * 60)
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run your experiment:")
        print("python scr/real_experiment.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error applying fixes: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

