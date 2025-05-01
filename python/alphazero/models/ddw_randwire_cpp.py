# python/alphazero/models/ddw_randwire_cpp.py
import torch
import torch.nn as nn
import os
import sys

# Add the build directory to path for importing C++ extension
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
build_dir = os.path.join(os.path.dirname(src_dir), 'build', 'src', 'pybind')
sys.path.insert(0, build_dir)

try:
    import _alphazero_cpp as az
except ImportError as e:
    print(f"Error importing C++ module: {e}")
    print("Make sure the project is properly built with Python bindings.")
    raise

class DDWRandWireResNetWrapper(nn.Module):
    """
    Python wrapper for the C++ implementation of DDWRandWireResNet.
    
    This wrapper makes the C++ model compatible with PyTorch's Python API
    while still using the native C++ implementation for maximum performance.
    """
    
    def __init__(self, input_channels, output_size, channels=128, num_blocks=20):
        """
        Initialize the DDWRandWireResNet model.
        
        Args:
            input_channels: Number of input channels
            output_size: Size of policy output
            channels: Number of channels in the network (default: 128)
            num_blocks: Number of random wire blocks (default: 20)
        """
        super(DDWRandWireResNetWrapper, self).__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        self.channels = channels
        self.num_blocks = num_blocks
        
        # Create the C++ model
        self.cpp_model = az.createDDWRandWireResNet(
            input_channels, output_size, channels, num_blocks
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_channels, height, width]
            
        Returns:
            Tuple of (policy, value) tensors
        """
        return self.cpp_model.forward(x)
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: File path to save the model
        """
        self.cpp_model.save(path)
    
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path: File path to load the model from
        """
        self.cpp_model.load(path)
    
    def export_to_torchscript(self, path, input_shape=None):
        """
        Export the model to TorchScript format for C++ inference.
        
        Args:
            path: File path to save the exported model
            input_shape: Optional input tensor shape [batch_size, channels, height, width]
        """
        if input_shape is None:
            input_shape = [1, self.input_channels, 0, 0]
        self.cpp_model.export_to_torchscript(path, input_shape)

def create_model(input_channels, output_size, channels=128, num_blocks=20):
    """
    Factory function to create a DDWRandWireResNetWrapper model.
    
    Args:
        input_channels: Number of input channels
        output_size: Size of policy output
        channels: Number of channels in the network (default: 128)
        num_blocks: Number of random wire blocks (default: 20)
        
    Returns:
        A DDWRandWireResNetWrapper model
    """
    return DDWRandWireResNetWrapper(input_channels, output_size, channels, num_blocks)

def load_model(path, input_channels, output_size, channels=128, num_blocks=20):
    """
    Factory function to load a DDWRandWireResNetWrapper model from a file.
    
    Args:
        path: File path to load the model from
        input_channels: Number of input channels
        output_size: Size of policy output
        channels: Number of channels in the network (default: 128)
        num_blocks: Number of random wire blocks (default: 20)
        
    Returns:
        A loaded DDWRandWireResNetWrapper model
    """
    model = DDWRandWireResNetWrapper(input_channels, output_size, channels, num_blocks)
    model.load(path)
    return model

def export_to_torchscript(model, path, input_shape=None):
    """
    Export a model to TorchScript format for C++ inference.
    
    Args:
        model: The model to export
        path: File path to save the exported model
        input_shape: Optional input tensor shape [batch_size, channels, height, width]
    """
    if isinstance(model, DDWRandWireResNetWrapper):
        model.export_to_torchscript(path, input_shape)
    else:
        # If it's a standard PyTorch model, use the standard export path
        if input_shape is None:
            input_shape = [1, model.input_channels, 15, 15]
        
        dummy_input = torch.zeros(input_shape)
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(path)

def test_model():
    """
    Test the C++ DDWRandWireResNet model with random input.
    """
    # Create a model with input_channels=18, output_size=225 (15x15 board)
    model = create_model(input_channels=18, output_size=225)
    
    # Create random input tensor
    x = torch.randn(4, 18, 15, 15)
    
    # Forward pass
    policy, value = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test export
    model.export_to_torchscript("test_model.pt")
    print(f"Model exported to test_model.pt")
    
    # Create another random model for comparison
    model2 = create_model(input_channels=18, output_size=225)
    
    # Compare if different random seeds generate different models
    x2 = torch.randn(1, 18, 15, 15)
    p1, v1 = model(x2)
    p2, v2 = model2(x2)
    
    # Check if outputs are different (should be for random initialization)
    diff_policy = torch.abs(p1 - p2).mean().item()
    diff_value = torch.abs(v1 - v2).mean().item()
    
    print(f"Average policy difference between models: {diff_policy}")
    print(f"Average value difference between models: {diff_value}")
    
    return model

if __name__ == "__main__":
    test_model()