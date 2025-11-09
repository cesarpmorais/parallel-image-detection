#!/usr/bin/env python3
"""
Validation Framework for C++ ResNet18 Implementation
Compares C++ outputs against Python reference at each layer
"""

import numpy as np
import os
import glob

class LayerValidator:
    def __init__(self, reference_dir='reference_outputs', cpp_dir='cpp_outputs'):
        self.reference_dir = reference_dir
        self.cpp_dir = cpp_dir
        self.tolerance = 1e-4
        
    def load_tensor(self, filepath, shape_file=None):
        """Load binary tensor and reshape"""
        if not os.path.exists(filepath):
            return None
            
        data = np.fromfile(filepath, dtype=np.float32)
        
        # Try to load shape
        if shape_file and os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape = tuple(map(int, f.read().strip().split()))
            data = data.reshape(shape)
        
        return data
    
    def compare_tensors(self, ref_tensor, cpp_tensor, layer_name):
        """Compare two tensors with detailed analysis"""
        if ref_tensor is None:
            print(f"{layer_name}: Reference tensor not found")
            return False
            
        if cpp_tensor is None:
            print(f"{layer_name}: C++ tensor not found")
            return False
        
        # Shape check
        if ref_tensor.shape != cpp_tensor.shape:
            print(f"{layer_name}: Shape mismatch")
            print(f"   Reference: {ref_tensor.shape}")
            print(f"   C++:       {cpp_tensor.shape}")
            return False
        
        # Value comparison
        diff = np.abs(ref_tensor - cpp_tensor)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Statistics
        ref_stats = {
            'min': np.min(ref_tensor),
            'max': np.max(ref_tensor),
            'mean': np.mean(ref_tensor),
            'std': np.std(ref_tensor)
        }
        
        cpp_stats = {
            'min': np.min(cpp_tensor),
            'max': np.max(cpp_tensor),
            'mean': np.mean(cpp_tensor),
            'std': np.std(cpp_tensor)
        }
        
        # Check tolerance
        passed = max_diff < self.tolerance
        
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {layer_name}:")
        print(f"   Shape: {ref_tensor.shape}")
        print(f"   Max diff: {max_diff:.2e}")
        print(f"   Mean diff: {mean_diff:.2e}")
        print(f"   Reference: min={ref_stats['min']:.4f}, max={ref_stats['max']:.4f}, mean={ref_stats['mean']:.4f}")
        print(f"   C++:       min={cpp_stats['min']:.4f}, max={cpp_stats['max']:.4f}, mean={cpp_stats['mean']:.4f}")
        
        if not passed:
            # Find worst mismatches
            worst_indices = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   Worst mismatch at {worst_indices}:")
            print(f"     Reference: {ref_tensor[worst_indices]:.6f}")
            print(f"     C++:       {cpp_tensor[worst_indices]:.6f}")
        
        return passed
    
    def validate_all_layers(self):
        """Validate all layers in sequence"""
        
        # Define layer sequence
        layers = [
            'input',
            'after_conv1',
            'after_bn1', 
            'after_relu1',
            'after_maxpool',
            'after_layer1',
            'after_layer2',
            'after_layer3',
            'after_layer4',
            'after_avgpool',
            'after_flatten',
            'final_output'
        ]
        
        print("Validating C++ implementation against Python reference...\n")
        
        passed_count = 0
        total_count = 0
        
        for layer in layers:
            ref_file = os.path.join(self.reference_dir, f"{layer}.bin")
            cpp_file = os.path.join(self.cpp_dir, f"{layer}.bin")
            shape_file = os.path.join(self.reference_dir, f"{layer}_shape.txt")
            
            ref_tensor = self.load_tensor(ref_file, shape_file)
            cpp_tensor = self.load_tensor(cpp_file, shape_file)
            
            if ref_tensor is not None or cpp_tensor is not None:
                total_count += 1
                if self.compare_tensors(ref_tensor, cpp_tensor, layer):
                    passed_count += 1
                print()
        
        # Summary
        print(f"Validation Summary:")
        print(f"   Passed: {passed_count}/{total_count}")
        print(f"   Success rate: {passed_count/total_count*100:.1f}%" if total_count > 0 else "   No tests run")
        
        return passed_count == total_count
    
    def validate_single_layer(self, layer_name):
        """Validate a single layer"""
        ref_file = os.path.join(self.reference_dir, f"{layer_name}.bin")
        cpp_file = os.path.join(self.cpp_dir, f"{layer_name}.bin")
        shape_file = os.path.join(self.reference_dir, f"{layer_name}_shape.txt")
        
        ref_tensor = self.load_tensor(ref_file, shape_file)
        cpp_tensor = self.load_tensor(cpp_file, shape_file)
        
        return self.compare_tensors(ref_tensor, cpp_tensor, layer_name)

def generate_test_data():
    """Generate simple test data for initial validation"""
    print("Generating test data...")
    
    # Create simple test input (1x3x224x224)
    np.random.seed(42)  # Reproducible
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    os.makedirs('test_data', exist_ok=True)
    test_input.tofile('test_data/test_input.bin')
    
    with open('test_data/test_input_shape.txt', 'w') as f:
        f.write(' '.join(map(str, test_input.shape)))
    
    print(f"Test input saved: {test_input.shape}")
    return test_input

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate C++ ResNet18 implementation')
    parser.add_argument('--layer', type=str, help='Validate specific layer only')
    parser.add_argument('--generate-test', action='store_true', help='Generate test data')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Comparison tolerance')
    
    args = parser.parse_args()
    
    if args.generate_test:
        generate_test_data()
        exit(0)
    
    validator = LayerValidator()
    validator.tolerance = args.tolerance
    
    if args.layer:
        validator.validate_single_layer(args.layer)
    else:
        validator.validate_all_layers()
