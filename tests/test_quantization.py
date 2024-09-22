import unittest
import torch
from transformers import AutoModelForSequenceClassification
from src.quantization import quantize_model
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def test_dynamic_quantization(self):
        quantized_model = quantize_model(self.model, quantization_type="dynamic")
        self.assertIsInstance(quantized_model, torch.nn.Module)
        self.assertTrue(any(isinstance(module, torch.nn.quantized.dynamic.Linear) for module in quantized_model.modules()))

    def test_static_quantization(self):
        quantized_model = quantize_model(self.model, quantization_type="static")
        self.assertIsInstance(quantized_model, torch.nn.Module)
        self.assertTrue(any(isinstance(module, torch.nn.quantized.Linear) for module in quantized_model.modules()))

    def test_gptq_quantization(self):
        quantized_model = quantize_model(self.model, quantization_type="gptq", model_name="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Check if the model is an instance of AutoGPTQForCausalLM
        self.assertIsInstance(quantized_model, AutoGPTQForCausalLM)
        
        # Check if the model has been quantized to 4-bit
        self.assertEqual(quantized_model.quantize_config.bits, 4)
        
        # Check if the group size is set correctly
        self.assertEqual(quantized_model.quantize_config.group_size, 128)
        
        # Verify that the model parameters are now in int32 format (typical for 4-bit quantization)
        for param in quantized_model.parameters():
            if param.requires_grad:
                self.assertEqual(param.dtype, torch.int32)
        
        # Check if the model can still perform inference
        input_ids = torch.randint(0, 1000, (1, 10))  # Random input
        with torch.no_grad():
            output = quantized_model(input_ids)
        self.assertIsNotNone(output)

    def test_awq_quantization(self):
        quantized_model = quantize_model(self.model, quantization_type="awq", model_name="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Check if the model is an instance of AutoAWQForCausalLM
        self.assertIsInstance(quantized_model, AutoAWQForCausalLM)
        
        # Check if the model has been quantized to 4-bit
        self.assertEqual(quantized_model.quantization_config.bits, 4)
        
        # Check if the group size is set correctly
        self.assertEqual(quantized_model.quantization_config.group_size, 128)
        
        # Verify that the model parameters are now in int8 format (typical for AWQ)
        for name, module in quantized_model.named_modules():
            if 'layer' in name:
                self.assertEqual(module.weight.dtype, torch.int8)
        
        # Check if the model can still perform inference
        input_ids = torch.randint(0, 1000, (1, 10))  # Random input
        with torch.no_grad():
            output = quantized_model(input_ids)
        self.assertIsNotNone(output)

    def test_invalid_quantization_type(self):
        with self.assertRaises(ValueError):
            quantize_model(self.model, quantization_type="invalid")

if __name__ == '__main__':
    unittest.main()