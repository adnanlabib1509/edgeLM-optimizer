# EdgeLM-Optimizer

EdgeLM-Optimizer is an project that demonstrates various optimization techniques for deploying large language models on edge devices. It showcases quantization methods, pruning, and simulated edge device inference to optimize model size and inference speed while maintaining acceptable performance.

## Project Structure

- `src/`: Contains the core functionality of the project
  - `model_loader.py`: Handles loading pre-trained models
  - `quantization.py`: Implements various model quantization techniques
  - `pruning.py`: Implements model pruning techniques
  - `edge_simulator.py`: Simulates edge device constraints
  - `benchmark.py`: Runs performance benchmarks
- `tests/`: Contains unit tests for the core functionality
- `main.py`: Main script to run the optimization pipeline
- `requirements.txt`: List of project dependencies

## Key Features

1. **Model Loading**: Utilizes Hugging Face's Transformers library to load pre-trained language models.

2. **Quantization Techniques**:
   - Dynamic Quantization: PyTorch's built-in dynamic quantization
   - Static Quantization: PyTorch's built-in static quantization
   - GPTQ (Generative Pre-trained Transformer Quantization): Advanced quantization specifically for transformer models
   - AWQ (Activation-aware Weight Quantization): Cutting-edge quantization considering activation statistics

3. **Pruning**: Applies weight pruning to remove less important connections in the model, further reducing its size.

4. **Edge Device Simulation**: Simulates edge device constraints such as limited CPU speed and memory to provide realistic performance estimates.

5. **Comprehensive Benchmarking**: Measures inference time and model size for various optimization techniques, allowing for easy comparison.

## How It Works

1. **Model Loading**: The project starts by loading a pre-trained DistilBERT model for sentiment analysis.

2. **Edge Device Simulation**: An EdgeDeviceSimulator is created to mimic the constraints of an edge device, including reduced CPU speed and limited memory.

3. **Optimization Techniques**:
   - Multiple quantization methods are applied (Dynamic, Static, GPTQ, AWQ)
   - Pruning is performed to reduce model size
   - Combined optimization (quantization + pruning) is also demonstrated

4. **Benchmarking**: Each version of the model is benchmarked on the simulated edge device, measuring:
   - Inference Time: The time taken to process a sample input
   - Model Size: The memory footprint of the model

5. **Results Comparison**: The main script outputs a comparison of the different optimization techniques, allowing for easy evaluation of their effectiveness.

## Usage

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Run the main optimization pipeline:

```
python main.py
```

3. Run the unit tests:

```
python -m unittest discover tests
```

## Extending the Project

- **Custom Models**: Modify `model_loader.py` to experiment with different pre-trained models.
- **Optimization Techniques**: Implement additional quantization or pruning methods in their respective files.
- **Edge Device Profiles**: Create different edge device profiles in `edge_simulator.py` to test various hardware constraints.
- **Benchmark Metrics**: Add more performance metrics in `benchmark.py` for comprehensive evaluation.

## Advanced Quantization Techniques

This project showcases two cutting-edge quantization techniques:

1. **GPTQ (Generative Pre-trained Transformer Quantization)**:
- Specifically designed for transformer models
- Achieves high compression rates while maintaining model quality
- Implemented using the `auto-gptq` library

2. **AWQ (Activation-aware Weight Quantization)**:
- Considers activation statistics during quantization
- Provides excellent balance between model size and performance
- Implemented using the `awq` library

These advanced techniques demonstrate the project's alignment with state-of-the-art practices in model optimization for edge devices.

## Testing

The project includes a comprehensive test suite covering:
- Quantization methods (including GPTQ and AWQ)
- Pruning functionality
- Benchmarking process
- Edge device simulation

Run the tests to ensure the reliability and correctness of the optimization techniques.

## License

This project is licensed under the MIT License.