"""
Neural network models for AlphaZero.
"""

from alphazero.models.ddw_randwire import DDWRandWireResNet
try:
    from alphazero.models.ddw_randwire_cpp import DDWRandWireResNetWrapper, create_model, load_model, export_to_torchscript
    __all__ = ['DDWRandWireResNet', 'DDWRandWireResNetWrapper', 'create_model', 'load_model', 'export_to_torchscript']
except ImportError:
    # C++ extension not available
    __all__ = ['DDWRandWireResNet']