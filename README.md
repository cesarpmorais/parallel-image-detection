# ResNet18 C++ Implementation with Step-by-Step Validation

Parallel computing final project implementing ResNet18 from scratch in C++ with custom parallelization.

## Project Structure

```
tp_final/
├── weights/                    # Extracted PyTorch model weights (64 files)
├── docs/                      # Documentation
│   ├── model.md              # ResNet18 architecture overview
│   └── implementation_guide.md # Detailed C++ implementation guide
├── reference_outputs/         # Python reference intermediate outputs
├── test_data/                # Test input data
├── cpp_outputs/              # C++ implementation outputs (to be created)
├── model_extraction.py       # Extract weights from PyTorch
├── reference_model.py        # Python reference with logging
├── validate.py              # Layer-by-layer validation framework
└── run_tests.py             # Test runner
```

## Quick Start

### 1. Generate Reference Outputs
```bash
python run_tests.py
```
This creates:
- `test_data/input.bin` - Test input for C++ implementation
- `reference_outputs/` - Python reference outputs for each layer

### 2. Implement C++ Layers
Follow the implementation guide in `docs/implementation_guide.md`:

1. Start with Conv2D layer
2. Add ReLU activation
3. Test against reference: `python validate.py --layer after_conv1`
4. Continue with BatchNorm, MaxPool, etc.

### 3. Validate Each Layer
```bash
# Validate specific layer
python validate.py --layer after_conv1

# Validate all layers
python validate.py

# Adjust tolerance if needed
python validate.py --tolerance 1e-3
```

### 4. C++ Output Format
Your C++ implementation should save outputs to `cpp_outputs/` as:
- Binary files: `after_conv1.bin`, `after_layer1.bin`, etc.
- Same format as reference: float32, row-major order
- Use reference shapes from `reference_outputs/*_shape.txt`

## Testing Strategy

### Layer-by-Layer Validation
1. **Conv1**: Test basic convolution implementation
2. **BatchNorm**: Validate normalization (or use fused weights)
3. **ReLU**: Simple activation function
4. **MaxPool**: Pooling operation
5. **Layer1-4**: Residual blocks with skip connections
6. **AvgPool**: Adaptive average pooling
7. **FC**: Final classification layer

### Test Input
- Simple synthetic input (not random) for reproducible testing
- Shape: 1×3×224×224 (single image, RGB, 224×224)
- Contains structured patterns for easier debugging

### Validation Metrics
- **Shape matching**: Exact tensor dimensions
- **Value comparison**: Absolute difference < 1e-4 (default)
- **Statistics**: Min, max, mean comparison
- **Error localization**: Shows worst mismatches

## Implementation Tips

### Memory Layout
- Use row-major (C-style) memory layout
- Match PyTorch tensor ordering: [batch, channels, height, width]
- Pre-allocate buffers for intermediate activations

### Debugging
- Start with single layer validation
- Use small tolerance initially (1e-6) then relax if needed
- Check intermediate statistics (min/max/mean) for sanity
- Compare against reference layer by layer

### Performance
- Focus on correctness first, optimization later
- Profile after full implementation is validated
- Consider SIMD, threading, memory optimization

## Dataset Integration

For Pascal VOC 2012 testing:
1. Place dataset in `dataset/` folder
2. Add test images to `dataset/test_images/`
3. Use `reference_model.py` with real images
4. Validate object detection pipeline

## Files Overview

- **model_extraction.py**: Extracts weights from PyTorch ResNet18
- **reference_model.py**: Python reference with intermediate logging
- **validate.py**: Compares C++ vs Python outputs
- **run_tests.py**: Orchestrates testing workflow
