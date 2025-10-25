#!/usr/bin/env python3
"""
Test Runner - Generate reference outputs and validate C++ implementation
"""

import torch
import numpy as np
import os
from reference_model import ResNet18Reference, preprocess_image
from validate import LayerValidator, generate_test_data

def create_simple_test_input():
    """Create a simple test input for initial validation"""
    print("🎲 Creating simple test input...")
    
    # Simple test: all zeros with a few non-zero values
    test_input = torch.zeros(1, 3, 224, 224)
    test_input[0, 0, 112, 112] = 1.0  # Center pixel in red channel
    test_input[0, 1, 100:124, 100:124] = 0.5  # Small square in green
    test_input[0, 2, :, :] = 0.1  # Low value across blue channel
    
    return test_input

def run_reference_generation():
    """Generate reference outputs using simple test input"""
    print("🔄 Generating reference outputs...")
    
    # Create reference model
    ref_model = ResNet18Reference()
    
    # Create test input
    test_input = create_simple_test_input()
    
    # Save test input for C++
    os.makedirs('test_data', exist_ok=True)
    test_np = test_input.numpy().astype(np.float32)
    test_np.tofile('test_data/input.bin')
    
    with open('test_data/input_shape.txt', 'w') as f:
        f.write(' '.join(map(str, test_np.shape)))
    
    print(f"💾 Test input saved: {test_np.shape}")
    
    # Generate reference outputs
    with torch.no_grad():
        output = ref_model.forward_with_logging(test_input)
    
    print("✅ Reference outputs generated!")
    return output

def validate_cpp_implementation():
    """Validate C++ implementation if outputs exist"""
    print("\n🔍 Checking for C++ outputs...")
    
    if not os.path.exists('cpp_outputs'):
        print("❌ cpp_outputs/ directory not found")
        print("💡 Run your C++ implementation first to generate outputs")
        return False
    
    validator = LayerValidator()
    return validator.validate_all_layers()

def main():
    print("🚀 ResNet18 Test Runner")
    print("=" * 50)
    
    # Step 1: Generate reference outputs
    output = run_reference_generation()
    
    # Show final prediction for sanity check
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    print(f"\n🎯 Reference model top 5 predictions:")
    for i in range(5):
        print(f"  Class {top5_idx[i].item()}: {top5_prob[i].item():.6f}")
    
    # Step 2: Validate C++ implementation (if available)
    success = validate_cpp_implementation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All validations passed!")
    else:
        print("⚠️  Some validations failed or C++ outputs not found")
    
    print("\n📋 Next steps:")
    print("1. Implement C++ layers following docs/implementation_guide.md")
    print("2. Save outputs to cpp_outputs/ directory")
    print("3. Run: python validate.py")
    print("4. Compare layer by layer until all pass")

if __name__ == "__main__":
    main()
