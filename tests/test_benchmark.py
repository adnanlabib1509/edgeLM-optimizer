import unittest
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.edge_simulator import EdgeDeviceSimulator
from src.benchmark import run_benchmark

class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.edge_device = EdgeDeviceSimulator(cpu_speed=1.0, memory_limit=1024)
        self.text = "This is a test sentence."

    def test_run_benchmark(self):
        benchmark_results = run_benchmark(self.model, self.tokenizer, self.text, self.edge_device)
        
        self.assertIn('inference_time', benchmark_results)
        self.assertIn('model_size', benchmark_results)
        self.assertIn('output', benchmark_results)
        
        self.assertGreater(benchmark_results['inference_time'], 0)
        self.assertGreater(benchmark_results['model_size'], 0)

    def test_memory_limit_exceeded(self):
        small_edge_device = EdgeDeviceSimulator(cpu_speed=1.0, memory_limit=1)  # Very small memory limit
        with self.assertRaises(MemoryError):
            run_benchmark(self.model, self.tokenizer, self.text, small_edge_device)

if __name__ == '__main__':
    unittest.main()