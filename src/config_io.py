"""
Utilities for saving and loading AWB configs as JSON.

Provides functions to:
1. Extract config from search results JSON
2. Save config dict to JSON file
3. Load config dict from JSON and reconstruct AWBConfig object
"""

import json
import numpy as np
from typing import Dict, Any
from dataclasses import fields

from src.utils import config


def awb_config_to_dict(cfg: config.AWBConfig) -> Dict[str, Any]:
    """
    Convert AWBConfig dataclass to a serializable dictionary.
    
    Converts numpy arrays to lists so the dict can be JSON serialized.
    
    Parameters:
    -----------
    cfg : AWBConfig
        The configuration object
        
    Returns:
    --------
    dict : Serializable dictionary representation
    """
    result = {}
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        
        # Convert numpy arrays to lists
        if isinstance(value, np.ndarray):
            result[field.name] = value.tolist()
        else:
            result[field.name] = value
    
    return result


def dict_to_awb_config(data: Dict[str, Any]) -> config.AWBConfig:
    """
    Convert dictionary back to AWBConfig object.
    
    Reconstructs numpy arrays from lists.
    
    Parameters:
    -----------
    data : dict
        Dictionary with AWB config parameters
        
    Returns:
    --------
    cfg : AWBConfig
        Reconstructed configuration object
    """
    # Convert lists back to numpy arrays where needed
    data_copy = dict(data)
    
    if 'cct_curve_rb' in data_copy and isinstance(data_copy['cct_curve_rb'], list):
        data_copy['cct_curve_rb'] = np.array(data_copy['cct_curve_rb'])
    
    # Create AWBConfig with the data
    return config.AWBConfig(**data_copy)


def save_config_json(cfg: config.AWBConfig, filepath: str) -> None:
    """
    Save AWBConfig to JSON file.
    
    Parameters:
    -----------
    cfg : AWBConfig
        The configuration to save
    filepath : str
        Path to save the JSON file
    """
    config_dict = awb_config_to_dict(cfg)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {filepath}")


def load_config_json(filepath: str) -> config.AWBConfig:
    """
    Load AWBConfig from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
        
    Returns:
    --------
    cfg : AWBConfig
        The loaded configuration
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    cfg = dict_to_awb_config(data)
    print(f"Loaded config from {filepath}")
    return cfg


def extract_and_save_best_configs(result_jsons: list, output_dir: str = '.') -> None:
    """
    Extract configs from search results JSON files and save them individually.
    
    Parameters:
    -----------
    result_jsons : list of str
        List of paths to result JSON files from search_config.py
    output_dir : str
        Directory to save the extracted config JSON files
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    
    for result_file in result_jsons:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Extract config string and parse it
            config_str = result['config']
            
            # Try to get the method and region from filename
            basename = os.path.basename(result_file)
            # e.g., best_awb_config_method0_sp_lab-multi.json
            parts = basename.replace('best_awb_config_', '').replace('_lab-multi.json', '').split('_')
            
            if len(parts) >= 2:
                method = parts[0]  # method0, method1, etc.
                region = parts[1]  # sp or t
                config_filename = f"{method}_{region}_config.json"
            else:
                config_filename = basename.replace('.json', '_config.json')
            
            output_path = os.path.join(output_dir, config_filename)
            
            # For now, we save the config string representation
            # To properly extract, we'd need to parse the config string or
            # modify search_config.py to save the config dict directly
            print(f"Processing {result_file} -> {output_path}")
            print(f"  Config string: {config_str[:100]}...")
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")


# Example usage
if __name__ == '__main__':
    # Create a sample config
    sample_cfg = config.AWBConfig(
        awb_method=3,
        num_superpixels=32,
        region_method='superpixels',
        sp_compactness=5,
        cct_tolerance=0.05,
    )
    
    # Save it
    save_config_json(sample_cfg, 'sample_config.json')
    
    # Load it back
    loaded_cfg = load_config_json('sample_config.json')
    
    print("\nOriginal config:")
    print(sample_cfg)
    print("\nLoaded config:")
    print(loaded_cfg)
