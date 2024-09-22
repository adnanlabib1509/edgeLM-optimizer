import torch
from .edge_simulator import EdgeDeviceSimulator

def run_benchmark(model, tokenizer, text, edge_device):
    # Prepare input
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Check if model fits in simulated device memory
    if not edge_device.check_memory_usage(model):
        raise MemoryError("Model exceeds simulated device memory limit")
    
    # Run inference and measure time
    output, inference_time = edge_device.simulate_inference(model, input_ids)
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Return benchmark results
    return {
        "inference_time": inference_time,
        "output": output,
        "model_size": model_size
    }