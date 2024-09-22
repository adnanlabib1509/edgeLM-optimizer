import torch
from src.model_loader import load_model
from src.quantization import quantize_model
from src.pruning import prune_model
from src.edge_simulator import EdgeDeviceSimulator
from src.benchmark import run_benchmark

def main():
    # Load model
    model, tokenizer = load_model()
    
    # Create edge device simulator
    edge_device = EdgeDeviceSimulator(cpu_speed=0.5, memory_limit=256)
    
    # Sample text for inference
    text = "This project demonstrates edge optimization techniques."
    
    # Benchmark original model
    print("Original Model:")
    original_benchmark = run_benchmark(model, tokenizer, text, edge_device)
    print(f"Inference Time: {original_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {original_benchmark['model_size']:.2f} MB")
    
    # Quantize model (PyTorch dynamic quantization)
    quantized_model = quantize_model(model, quantization_type="dynamic")
    print("\nDynamically Quantized Model:")
    quantized_benchmark = run_benchmark(quantized_model, tokenizer, text, edge_device)
    print(f"Inference Time: {quantized_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {quantized_benchmark['model_size']:.2f} MB")
    
    # GPTQ quantized model
    gptq_model = quantize_model(model, quantization_type="gptq", model_name="gpt2")
    print("\nGPTQ Quantized Model:")
    gptq_benchmark = run_benchmark(gptq_model, tokenizer, text, edge_device)
    print(f"Inference Time: {gptq_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {gptq_benchmark['model_size']:.2f} MB")

    # AWQ quantized model
    awq_model = quantize_model(model, quantization_type="awq", model_name="gpt2")
    print("\nAWQ Quantized Model:")
    awq_benchmark = run_benchmark(awq_model, tokenizer, text, edge_device)
    print(f"Inference Time: {awq_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {awq_benchmark['model_size']:.2f} MB")
    
    # Prune model
    pruned_model = prune_model(model)
    print("\nPruned Model:")
    pruned_benchmark = run_benchmark(pruned_model, tokenizer, text, edge_device)
    print(f"Inference Time: {pruned_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {pruned_benchmark['model_size']:.2f} MB")
    
    # Quantize and prune
    optimized_model = quantize_model(prune_model(model))
    print("\nOptimized Model (Quantized + Pruned):")
    optimized_benchmark = run_benchmark(optimized_model, tokenizer, text, edge_device)
    print(f"Inference Time: {optimized_benchmark['inference_time']:.2f} ms")
    print(f"Model Size: {optimized_benchmark['model_size']:.2f} MB")

if __name__ == "__main__":
    main()