import torch

class EdgeDeviceSimulator:
    def __init__(self, cpu_speed=1.0, memory_limit=512):
        self.cpu_speed = cpu_speed  # Simulated CPU speed multiplier
        self.memory_limit = memory_limit  # Simulated memory limit in MB

    def check_memory_usage(self, model):
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        return model_size <= self.memory_limit

    def simulate_inference(self, model, input_ids):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            output = model(input_ids)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / self.cpu_speed
        
        return output, elapsed_time