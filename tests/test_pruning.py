import unittest
import torch
from transformers import AutoModelForSequenceClassification
from src.pruning import prune_model

class TestPruning(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def test_pruning(self):
        original_num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pruned_model = prune_model(self.model, amount=0.3)
        pruned_num_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        
        self.assertLess(pruned_num_params, original_num_params)
        self.assertGreater(pruned_num_params, 0.6 * original_num_params)  # Roughly 30% reduction

    def test_pruning_amount(self):
        with self.assertRaises(ValueError):
            prune_model(self.model, amount=1.5)  # Invalid pruning amount

if __name__ == '__main__':
    unittest.main()