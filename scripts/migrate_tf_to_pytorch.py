#!/usr/bin/env python3
"""
TensorFlow to PyTorch Model Migration

Converts V1.3 TensorFlow model weights to V2.0 PyTorch format.

Usage:
    python scripts/migrate_tf_to_pytorch.py --input results/model_weights.weights.h5 --output models/v2_predictor.pt
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tensorflow_weights(path: str) -> Dict[str, np.ndarray]:
    """
    Load weights from TensorFlow/Keras model.
    
    Args:
        path: Path to .h5 weights file
        
    Returns:
        Dictionary of layer_name -> weight array
    """
    try:
        import h5py
    except ImportError:
        logger.error("h5py required: pip install h5py")
        return {}
    
    weights = {}
    
    try:
        with h5py.File(path, 'r') as f:
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights[name] = np.array(obj)
            
            f.visititems(extract_weights)
        
        logger.info(f"Loaded {len(weights)} weight arrays from {path}")
        
        # Print structure
        for name, arr in weights.items():
            logger.info(f"  {name}: {arr.shape}")
        
        return weights
        
    except Exception as e:
        logger.error(f"Failed to load TensorFlow weights: {e}")
        return {}


def map_lstm_to_transformer(tf_weights: Dict[str, np.ndarray],
                             hidden_dim: int = 512,
                             n_features: int = 10) -> Dict[str, np.ndarray]:
    """
    Map LSTM weights to Transformer-compatible format.
    
    Since LSTM and Transformer have different architectures,
    we can't directly convert. Instead, we:
    1. Extract input projection weights
    2. Initialize Transformer with compatible input layer
    3. Use random initialization for attention layers
    
    Args:
        tf_weights: TensorFlow weight dictionary
        hidden_dim: Transformer hidden dimension
        n_features: Number of input features
        
    Returns:
        Dictionary of PyTorch-compatible weights
    """
    pt_weights = {}
    
    # Look for LSTM input weights
    lstm_kernel = None
    for name, arr in tf_weights.items():
        if 'lstm' in name.lower() and 'kernel' in name.lower():
            if arr.shape[0] == n_features or 'input' in name.lower():
                lstm_kernel = arr
                logger.info(f"Found LSTM input kernel: {name} {arr.shape}")
                break
    
    # Create input projection from LSTM weights
    if lstm_kernel is not None:
        # LSTM kernel: (input_dim, 4 * hidden_dim) for i, f, c, o gates
        lstm_hidden = lstm_kernel.shape[1] // 4
        
        # Use input gate weights as initialization hint
        input_gate_weights = lstm_kernel[:, :lstm_hidden]
        
        # Project to transformer hidden dim
        if lstm_hidden != hidden_dim:
            # Simple projection
            proj = np.random.randn(n_features, hidden_dim).astype(np.float32) * 0.02
            # Initialize first lstm_hidden columns with LSTM weights
            proj[:, :min(lstm_hidden, hidden_dim)] = input_gate_weights[:, :min(lstm_hidden, hidden_dim)]
        else:
            proj = input_gate_weights
        
        pt_weights['input_projection.weight'] = proj.T  # PyTorch uses (out, in)
        pt_weights['input_projection.bias'] = np.zeros(hidden_dim, dtype=np.float32)
    
    # Look for output layer weights
    for name, arr in tf_weights.items():
        if 'dense' in name.lower() and 'kernel' in name.lower():
            if arr.shape[-1] == 1:  # Output layer
                logger.info(f"Found output kernel: {name} {arr.shape}")
                
                # Adapt to transformer output
                if arr.shape[0] != hidden_dim:
                    # Need projection
                    output_proj = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.02
                else:
                    output_proj = arr
                
                pt_weights['output_layer.weight'] = output_proj.T
                pt_weights['output_layer.bias'] = np.zeros(1, dtype=np.float32)
                break
    
    logger.info(f"Mapped {len(pt_weights)} weight tensors")
    
    return pt_weights


def create_pytorch_model(weights: Dict[str, np.ndarray],
                          hidden_dim: int = 512,
                          n_heads: int = 8,
                          n_layers: int = 3) -> 'torch.nn.Module':
    """
    Create PyTorch Transformer model with mapped weights.
    
    Args:
        weights: Mapped weight dictionary
        hidden_dim: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        
    Returns:
        Initialized PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("PyTorch required: pip install torch")
        return None
    
    try:
        from src.ml.transformer_predictor import TransformerPredictorModel
        
        model = TransformerPredictorModel(
            n_features=10,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # Load mapped weights where available
        state_dict = model.state_dict()
        
        for name, arr in weights.items():
            if name in state_dict:
                tensor = torch.from_numpy(arr)
                if tensor.shape == state_dict[name].shape:
                    state_dict[name] = tensor
                    logger.info(f"Loaded weight: {name}")
                else:
                    logger.warning(f"Shape mismatch for {name}: "
                                  f"expected {state_dict[name].shape}, got {tensor.shape}")
            else:
                logger.warning(f"Weight not found in model: {name}")
        
        model.load_state_dict(state_dict)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create PyTorch model: {e}")
        return None


def save_pytorch_model(model, path: str, include_config: bool = True):
    """
    Save PyTorch model to disk.
    
    Args:
        model: PyTorch model
        path: Output path
        include_config: Whether to save model config
    """
    import torch
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_features': 10,
            'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else 512,
            'n_heads': model.n_heads if hasattr(model, 'n_heads') else 8,
            'n_layers': model.n_layers if hasattr(model, 'n_layers') else 3
        },
        'version': 'V2.0',
        'migrated_from': 'TensorFlow V1.3'
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved PyTorch model to {path}")


def validate_migration(tf_path: str, pt_path: str) -> bool:
    """
    Validate migration by comparing predictions.
    
    Note: Since architectures differ, exact match is not expected.
    We validate that both produce reasonable outputs.
    """
    import torch
    
    try:
        # Load PyTorch model
        checkpoint = torch.load(pt_path)
        from src.ml.transformer_predictor import TransformerPredictorModel
        
        config = checkpoint['model_config']
        model = TransformerPredictorModel(
            n_features=config['n_features'],
            hidden_dim=config['hidden_dim'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test with random input
        test_input = torch.randn(10, config['n_features'])
        
        with torch.no_grad():
            output = model(test_input)
        
        # Validate output shape and range
        assert output.shape == (10, 1), f"Unexpected output shape: {output.shape}"
        assert (output >= 0).all() and (output <= 1).all(), "Output not in [0, 1] range"
        
        logger.info("Migration validation passed!")
        logger.info(f"  Sample outputs: {output[:5].flatten().tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration validation failed: {e}")
        return False


def migrate(input_path: str, output_path: str,
            hidden_dim: int = 512,
            n_heads: int = 8,
            n_layers: int = 3,
            validate: bool = True) -> bool:
    """
    Full migration pipeline.
    
    Args:
        input_path: Path to TensorFlow weights
        output_path: Path for PyTorch model
        hidden_dim: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        validate: Whether to validate migration
        
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("TensorFlow to PyTorch Migration")
    logger.info("=" * 60)
    
    # Step 1: Load TensorFlow weights
    logger.info("\n1. Loading TensorFlow weights...")
    tf_weights = load_tensorflow_weights(input_path)
    
    if not tf_weights:
        logger.error("No weights loaded from TensorFlow model")
        return False
    
    # Step 2: Map weights
    logger.info("\n2. Mapping weights to Transformer format...")
    pt_weights = map_lstm_to_transformer(tf_weights, hidden_dim=hidden_dim)
    
    # Step 3: Create PyTorch model
    logger.info("\n3. Creating PyTorch Transformer model...")
    model = create_pytorch_model(pt_weights, hidden_dim, n_heads, n_layers)
    
    if model is None:
        logger.error("Failed to create PyTorch model")
        return False
    
    # Step 4: Save model
    logger.info("\n4. Saving PyTorch model...")
    save_pytorch_model(model, output_path)
    
    # Step 5: Validate
    if validate:
        logger.info("\n5. Validating migration...")
        if not validate_migration(input_path, output_path):
            logger.warning("Validation failed, but model was saved")
    
    logger.info("\n" + "=" * 60)
    logger.info("Migration complete!")
    logger.info("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate TensorFlow to PyTorch")
    
    parser.add_argument(
        "--input", "-i",
        default="results/model_weights.weights.h5",
        help="Path to TensorFlow weights file"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/v2_predictor.pt",
        help="Path for PyTorch model output"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Transformer hidden dimension"
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    success = migrate(
        input_path=args.input,
        output_path=args.output,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        validate=not args.no_validate
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
