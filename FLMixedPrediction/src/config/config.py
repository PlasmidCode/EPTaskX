import yaml
import os
from typing import Dict, Any

class ConfigManager:
    """Configuration manager for FL+MoE renewable energy forecasting"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file path"""
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: str = None) -> None:
        """Save configuration to YAML file"""
        if save_path is None:
            save_path = self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_federated_config(self) -> Dict[str, Any]:
        """Get federated learning configuration"""
        return self.config.get('federated', {})
    
    def get_moe_config(self) -> Dict[str, Any]:
        """Get MoE configuration"""
        return self.config.get('moe', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})

# 全局配置实例
CONFIG = None

def init_config(config_path: str) -> ConfigManager:
    """Initialize global configuration"""
    global CONFIG
    CONFIG = ConfigManager(config_path)
    return CONFIG

def get_config() -> ConfigManager:
    """Get global configuration"""
    if CONFIG is None:
        raise RuntimeError("Configuration not initialized. Call init_config first.")
    return CONFIG
