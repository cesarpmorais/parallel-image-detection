# Validation Tutorial: Testing Your C++ ResNet18 Implementation

## Prerequisites

Your C++ implementation should:
1. **Read from `dataset/` folder** - All input images and data
2. **Generate these validation outputs**:
   ```
   cpp_outputs/
   ├── after_conv1.bin
   ├── after_bn1.bin  
   ├── after_relu1.bin
   ├── after_maxpool.bin
   ├── after_layer1.bin
   ├── after_layer2.bin
   ├── after_layer3.bin
   ├── after_layer4.bin
   ├── after_avgpool.bin
   ├── after_flatten.bin
   └── final_output.bin
   ```
3. **Use float32 format** - 4 bytes per value, row-major layout

## Step-by-Step Testing Guide

### Step 1: Generate Reference Data
```bash
cd src/validate_results
python run_tests.py
```
This creates:
- `test_data/input.bin` - Copy this to your `dataset/` folder
- `reference_outputs/` - Python reference for comparison

### Step 2: Test Each Layer Individually

#### 2.1 Test Conv1 Layer
```bash
# Run your C++ implementation (should read from dataset/)
./your_cpp_program

# Validate conv1 output
cd src/validate_results
python validate.py --layer after_conv1
```

**Expected output:**
```
✅ after_conv1:
   Shape: (1, 64, 112, 112)
   Max diff: 1.23e-05
   Mean diff: 2.45e-06
```

#### 2.2 Test BatchNorm + ReLU
```bash
python validate.py --layer after_bn1
python validate.py --layer after_relu1
```

#### 2.3 Test MaxPool
```bash
python validate.py --layer after_maxpool
```

#### 2.4 Test Residual Layers
```bash
python validate.py --layer after_layer1
python validate.py --layer after_layer2
python validate.py --layer after_layer3
python validate.py --layer after_layer4
```

#### 2.5 Test Final Layers
```bash
python validate.py --layer after_avgpool
python validate.py --layer after_flatten
python validate.py --layer final_output
```

### Step 3: Test All Layers at Once
```bash
python validate.py
```

**Expected output:**
```
🔍 Validating C++ implementation against Python reference...

✅ after_conv1:
   Shape: (1, 64, 112, 112)
   Max diff: 1.23e-05
   Mean diff: 2.45e-06

✅ after_bn1:
   Shape: (1, 64, 112, 112)
   Max diff: 8.91e-06
   Mean diff: 1.12e-06

... (all layers)

📊 Validation Summary:
   Passed: 11/11
   Success rate: 100.0%
```

## Debugging Failed Validations

### Shape Mismatch
```
❌ after_conv1:
   Shape mismatch
   Reference: (1, 64, 112, 112)
   C++:       (1, 64, 224, 224)
```
**Fix**: Check stride and padding in convolution

### Value Mismatch
```
❌ after_conv1:
   Shape: (1, 64, 112, 112)
   Max diff: 1.23e-01
   Mean diff: 2.45e-02
   Worst mismatch at (0, 5, 10, 15):
     Reference: 0.123456
     C++:       0.234567
```
**Fix**: Check weight loading, convolution math, or data types

### Missing Output
```
❌ after_conv1: C++ tensor not found
```
**Fix**: Ensure your C++ saves to `cpp_outputs/after_conv1.bin`

## Tolerance Adjustment

If you get small differences due to optimization:
```bash
# Relax tolerance to 1e-3
python validate.py --tolerance 1e-3

# Very relaxed for heavily optimized code
python validate.py --tolerance 1e-2
```

## Implementation Checklist

- [ ] C++ reads test input from `dataset/input.bin`
- [ ] C++ loads weights from `weights/*.bin` files
- [ ] C++ saves intermediate outputs to `cpp_outputs/`
- [ ] All outputs use float32 format
- [ ] Memory layout is row-major [batch, channels, height, width]
- [ ] Conv1 validation passes
- [ ] BatchNorm validation passes
- [ ] All residual layers validate
- [ ] Final output matches reference

## Quick Commands Reference

```bash
# Generate reference data
cd src/validate_results && python run_tests.py

# Test single layer
python validate.py --layer after_conv1

# Test all layers
python validate.py

# Adjust tolerance
python validate.py --tolerance 1e-3

# Check what outputs exist
ls ../cpp_outputs/
```

## Success Criteria

Your implementation is correct when:
1. All 11 layers pass validation
2. Max difference < 1e-4 (default tolerance)
3. Final output produces reasonable classification scores
4. No shape mismatches or missing files
