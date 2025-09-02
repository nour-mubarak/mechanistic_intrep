
"""
Basic tests for the Enhanced Mechanistic Interpretability System
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the system."""
    
    def test_imports(self):
        """Test that all main modules can be imported."""
        try:
            import activation_extractor
            import circuit_discovery
            import comprehensive_evaluation
            import enhanced_fine_tuning
            import interventions
            import real_experiment
            import visualizations
            import wandb_integration
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")
    
    def test_config_classes(self):
        """Test that configuration classes can be instantiated."""
        from comprehensive_evaluation import EvaluationConfig
        from enhanced_fine_tuning import BiasAwareTrainingConfig, AdvancedLoRAConfig
        from interventions import InterventionConfig
        from real_experiment import ExperimentConfig
        
        # Test instantiation
        eval_config = EvaluationConfig()
        bias_config = BiasAwareTrainingConfig()
        lora_config = AdvancedLoRAConfig()
        intervention_config = InterventionConfig(
            method="concept_erasure",
            target_layers=[1, 2, 3],
            strength=0.5,
            direction="suppress"
        )
        experiment_config = ExperimentConfig()
        
        # Basic assertions
        self.assertIsInstance(eval_config.significance_level, float)
        self.assertIsInstance(bias_config.bias_loss_weight, float)
        self.assertIsInstance(lora_config.base_rank, int)
        self.assertEqual(intervention_config.method, "concept_erasure")
        self.assertEqual(experiment_config.total_samples, 1000)

if __name__ == '__main__':
    unittest.main()

