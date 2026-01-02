# """
# Configuration Management Module

# This module handles loading and validating configuration files using Pydantic.
# Pydantic provides automatic validation, type checking, and clear error messages.
# """

# import os
# import yaml
# from pathlib import Path
# from typing import List, Dict, Optional, Literal
# from pydantic import BaseModel, Field, validator, field_validator
# from datetime import datetime


# # ============================================================================
# # Configuration Models - Using Pydantic for validation
# # ============================================================================

# class ProjectConfig(BaseModel):
#     """Project metadata configuration"""
#     name: str
#     version: str
#     description: str


# class DataGenerationConfig(BaseModel):
#     """Data generation parameters"""
#     start_date: str
#     end_date: str
#     num_stores: int = Field(gt=0, description="Must be positive")
#     num_items: int = Field(gt=0, description="Must be positive")
#     seasonality_strength: float = Field(ge=0, le=1)
#     trend_strength: float = Field(ge=0, le=1)
#     noise_level: float = Field(ge=0, le=1)
#     anomaly_percentage: float = Field(ge=0, le=1)
    
#     @field_validator('start_date', 'end_date')
#     @classmethod
#     def validate_date_format(cls, v):
#         """Ensure dates are in correct format"""
#         try:
#             datetime.strptime(v, '%Y-%m-%d')
#             return v
#         except ValueError:
#             raise ValueError(f"Date must be in YYYY-MM-DD format, got {v}")


# class DataConfig(BaseModel):
#     """Data paths and generation configuration"""
#     raw_dir: str
#     raw_sales: str
#     raw_calendar: str
#     processed_dir: str
#     processed_data: str
#     validation_dir: str
#     generation: DataGenerationConfig
    
#     @field_validator('raw_dir', 'processed_dir', 'validation_dir')
#     @classmethod
#     def validate_directory_path(cls, v):
#         """Ensure directory paths don't have trailing slashes"""
#         return v.rstrip('/')


# class FeatureConfig(BaseModel):
#     """Feature engineering configuration"""
#     lag_windows: List[int]
#     rolling_windows: List[int]
#     rolling_stats: List[str]
#     extract_date_features: bool
#     cyclical_encoding: bool
#     use_feature_selection: bool
#     selection_method: Literal["mutual_info", "correlation", "model_based"]
#     max_features: int = Field(gt=0)
#     train_size: float = Field(gt=0, lt=1)
#     val_size: float = Field(gt=0, lt=1)
#     test_size: float = Field(gt=0, lt=1)
    
#     @field_validator('train_size', 'val_size', 'test_size')
#     @classmethod
#     def validate_split_sum(cls, v, info):
#         """Ensure train+val+test sums to 1.0"""
#         # This is a simplified check - full validation would need all three values
#         if v <= 0 or v >= 1:
#             raise ValueError("Split sizes must be between 0 and 1")
#         return v


# class XGBoostConfig(BaseModel):
#     """XGBoost model hyperparameters"""
#     n_estimators: int = Field(gt=0)
#     max_depth: int = Field(gt=0)
#     learning_rate: float = Field(gt=0, le=1)
#     subsample: float = Field(gt=0, le=1)
#     colsample_bytree: float = Field(gt=0, le=1)
#     min_child_weight: int = Field(ge=0)
#     gamma: float = Field(ge=0)
#     objective: str
#     random_state: int
#     n_jobs: int


# class IsolationForestConfig(BaseModel):
#     """Isolation Forest hyperparameters"""
#     n_estimators: int = Field(gt=0)
#     max_samples: int = Field(gt=0)
#     contamination: float = Field(gt=0, lt=0.5)
#     random_state: int
#     n_jobs: int


# class TuningConfig(BaseModel):
#     """Hyperparameter tuning configuration"""
#     enabled: bool
#     n_trials: int = Field(gt=0)
#     timeout: int = Field(gt=0)


# class ModelConfig(BaseModel):
#     """Model configuration"""
#     forecasting_model: Literal["xgboost", "lightgbm", "random_forest"]
#     anomaly_model: Literal["isolation_forest", "autoencoder"]
#     xgboost: XGBoostConfig
#     isolation_forest: IsolationForestConfig
#     tuning: TuningConfig


# class MLflowRegistryConfig(BaseModel):
#     """MLflow model registry configuration"""
#     staging_alias: str
#     production_alias: str


# class MLflowConfig(BaseModel):
#     """MLflow configuration"""
#     tracking_uri: str
#     experiment_name: str
#     artifact_location: str
#     backend_store_uri: str
#     registry: MLflowRegistryConfig


# class RateLimitConfig(BaseModel):
#     """API rate limiting configuration"""
#     enabled: bool
#     requests_per_minute: int = Field(gt=0)


# class CORSConfig(BaseModel):
#     """CORS configuration"""
#     enabled: bool
#     allow_origins: List[str]
#     allow_methods: List[str]
#     allow_headers: List[str]


# class APIConfig(BaseModel):
#     """API server configuration"""
#     host: str
#     port: int = Field(gt=0, lt=65536)
#     reload: bool
#     workers: int = Field(gt=0)
#     timeout: int = Field(gt=0)
#     rate_limit: RateLimitConfig
#     cors: CORSConfig


# class PrometheusConfig(BaseModel):
#     """Prometheus monitoring configuration"""
#     enabled: bool
#     port: int = Field(gt=0, lt=65536)


# class ThresholdsConfig(BaseModel):
#     """Performance thresholds for monitoring"""
#     max_latency_ms: int = Field(gt=0)
#     max_error_rate: float = Field(ge=0, le=1)
#     min_accuracy: float = Field(ge=0, le=1)


# class DriftConfig(BaseModel):
#     """Data drift detection configuration"""
#     enabled: bool
#     check_interval_hours: int = Field(gt=0)
#     drift_threshold: float = Field(ge=0, le=1)
#     reference_window_days: int = Field(gt=0)


# class AlertsConfig(BaseModel):
#     """Alerting configuration"""
#     enabled: bool
#     email_enabled: bool
#     slack_enabled: bool


# class MonitoringConfig(BaseModel):
#     """Monitoring and alerting configuration"""
#     prometheus: PrometheusConfig
#     collect_predictions: bool
#     collect_latency: bool
#     collect_errors: bool
#     thresholds: ThresholdsConfig
#     drift: DriftConfig
#     alerts: AlertsConfig


# class TimeBasedRetrainingConfig(BaseModel):
#     """Time-based retraining trigger"""
#     enabled: bool
#     interval_days: int = Field(gt=0)


# class PerformanceBasedRetrainingConfig(BaseModel):
#     """Performance-based retraining trigger"""
#     enabled: bool
#     rmse_threshold_increase: float = Field(gt=0)


# class DriftBasedRetrainingConfig(BaseModel):
#     """Drift-based retraining trigger"""
#     enabled: bool
#     num_drifted_features_threshold: int = Field(gt=0)


# class DataVolumeRetrainingConfig(BaseModel):
#     """Data volume-based retraining trigger"""
#     enabled: bool
#     min_new_samples: int = Field(gt=0)


# class RetrainingConfig(BaseModel):
#     """Automated retraining configuration"""
#     time_based: TimeBasedRetrainingConfig
#     performance_based: PerformanceBasedRetrainingConfig
#     drift_based: DriftBasedRetrainingConfig
#     data_volume: DataVolumeRetrainingConfig


# class SQLiteConfig(BaseModel):
#     """SQLite database configuration"""
#     path: str


# class PostgreSQLConfig(BaseModel):
#     """PostgreSQL database configuration"""
#     host: str
#     port: int = Field(gt=0, lt=65536)
#     database: str
#     username: str


# class DatabaseConfig(BaseModel):
#     """Database configuration"""
#     type: Literal["sqlite", "postgresql", "mysql"]
#     sqlite: SQLiteConfig
#     postgresql: PostgreSQLConfig


# class LoggingConfig(BaseModel):
#     """Logging configuration"""
#     level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
#     format: Literal["json", "text"]
#     log_dir: str
#     log_file: str
#     rotation: str
#     retention: str


# class TestingConfig(BaseModel):
#     """Testing configuration"""
#     test_data_size: int = Field(gt=0)
#     random_seed: int


# class DockerConfig(BaseModel):
#     """Docker configuration"""
#     image_name: str
#     image_tag: str


# class KubernetesConfig(BaseModel):
#     """Kubernetes deployment configuration"""
#     namespace: str
#     replicas: int = Field(gt=0)
#     cpu_request: str
#     cpu_limit: str
#     memory_request: str
#     memory_limit: str


# class DeploymentConfig(BaseModel):
#     """Deployment configuration"""
#     environment: Literal["development", "staging", "production"]
#     docker: DockerConfig
#     kubernetes: KubernetesConfig


# # ============================================================================
# # Main Configuration Class
# # ============================================================================

# class Config(BaseModel):
#     """
#     Main configuration class that combines all sub-configurations.
#     This is the single source of truth for all application settings.
#     """
#     project: ProjectConfig
#     data: DataConfig
#     features: FeatureConfig
#     model: ModelConfig
#     mlflow: MLflowConfig
#     api: APIConfig
#     monitoring: MonitoringConfig
#     retraining: RetrainingConfig
#     database: DatabaseConfig
#     logging: LoggingConfig
#     testing: TestingConfig
#     deployment: DeploymentConfig
    
#     class Config:
#         """Pydantic configuration"""
#         validate_assignment = True  # Validate on assignment after creation
#         extra = "forbid"  # Raise error if extra fields provided


# # ============================================================================
# # Configuration Loading Functions
# # ============================================================================

# def get_project_root() -> Path:
#     """
#     Get the project root directory.
#     Searches for the directory containing config/ folder.
#     """
#     current = Path(__file__).resolve()
#     # Go up until we find the project root (contains config/ folder)
#     for parent in [current] + list(current.parents):
#         if (parent / "config").exists():
#             return parent
#     raise RuntimeError("Could not find project root directory")


# def load_yaml_file(file_path: Path) -> Dict:
#     """
#     Load a YAML file and return its contents as a dictionary.
    
#     Args:
#         file_path: Path to the YAML file
        
#     Returns:
#         Dictionary containing YAML contents
        
#     Raises:
#         FileNotFoundError: If file doesn't exist
#         yaml.YAMLError: If YAML is invalid
#     """
#     if not file_path.exists():
#         raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
#     with open(file_path, 'r') as f:
#         try:
#             data = yaml.safe_load(f)
#             return data
#         except yaml.YAMLError as e:
#             raise ValueError(f"Invalid YAML in {file_path}: {e}")


# def load_config(config_path: Optional[str] = None) -> Config:
#     """
#     Load and validate the main configuration file.
    
#     Args:
#         config_path: Optional path to config file. If None, uses default location.
        
#     Returns:
#         Validated Config object
        
#     Raises:
#         FileNotFoundError: If config file not found
#         ValidationError: If config validation fails
        
#     Example:
#         >>> config = load_config()
#         >>> print(config.project.name)
#         'supply-chain-mlops'
#     """
#     if config_path is None:
#         root = get_project_root()
#         config_path = root / "config" / "config.yaml"
#     else:
#         config_path = Path(config_path)
    
#     print(f"Loading configuration from: {config_path}")
    
#     # Load YAML file
#     config_dict = load_yaml_file(config_path)
    
#     # Validate and create Config object
#     try:
#         config = Config(**config_dict)
#         print("✅ Configuration loaded and validated successfully!")
#         return config
#     except Exception as e:
#         print(f"❌ Configuration validation failed!")
#         print(f"Error: {e}")
#         raise


# def load_secrets(secrets_path: Optional[str] = None) -> Dict:
#     """
#     Load secrets from secrets.yaml file.
    
#     Args:
#         secrets_path: Optional path to secrets file
        
#     Returns:
#         Dictionary containing secrets
        
#     Note:
#         Returns empty dict if secrets file doesn't exist (for development)
#     """
#     if secrets_path is None:
#         root = get_project_root()
#         secrets_path = root / "config" / "secrets.yaml"
#     else:
#         secrets_path = Path(secrets_path)
    
#     if not secrets_path.exists():
#         print(f"⚠️  Secrets file not found: {secrets_path}")
#         print("Using default/empty secrets (OK for development)")
#         return {}
    
#     print(f"Loading secrets from: {secrets_path}")
#     secrets = load_yaml_file(secrets_path)
#     print("✅ Secrets loaded successfully!")
#     return secrets


# def create_directories_from_config(config: Config) -> None:
#     """
#     Create all directories specified in the configuration if they don't exist.
    
#     Args:
#         config: Config object
#     """
#     root = get_project_root()
    
#     directories = [
#         config.data.raw_dir,
#         config.data.processed_dir,
#         config.data.validation_dir,
#         config.logging.log_dir,
#         "models/registry",
#         "models/staging",
#         "models/production",
#     ]
    
#     for directory in directories:
#         dir_path = root / directory
#         dir_path.mkdir(parents=True, exist_ok=True)
    
#     print("✅ All required directories created!")


# # ============================================================================
# # Convenience Functions
# # ============================================================================

# def get_config() -> Config:
#     """
#     Get the configuration object (singleton pattern).
#     Loads config on first call, returns cached version afterward.
#     """
#     if not hasattr(get_config, "_config"):
#         get_config._config = load_config()
#         create_directories_from_config(get_config._config)
#     return get_config._config


# def get_secrets() -> Dict:
#     """
#     Get the secrets dictionary (singleton pattern).
#     """
#     if not hasattr(get_secrets, "_secrets"):
#         get_secrets._secrets = load_secrets()
#     return get_secrets._secrets


# # ============================================================================
# # Testing and Validation
# # ============================================================================

# if __name__ == "__main__":
#     """
#     Test the configuration loading.
#     Run this file directly to validate your configuration:
    
#     python src/config/config.py
#     """
#     print("=" * 70)
#     print("Configuration Loading Test")
#     print("=" * 70)
#     print()
    
#     try:
#         # Load configuration
#         config = load_config()
        
#         # Print some key settings
#         print("\n📋 Key Configuration Settings:")
#         print(f"  Project: {config.project.name} v{config.project.version}")
#         print(f"  Environment: {config.deployment.environment}")
#         print(f"  Model: {config.model.forecasting_model}")
#         print(f"  API Port: {config.api.port}")
#         print(f"  MLflow: {config.mlflow.tracking_uri}")
        
#         # Print data generation parameters
#         print("\n📊 Data Generation:")
#         print(f"  Date Range: {config.data.generation.start_date} to {config.data.generation.end_date}")
#         print(f"  Stores: {config.data.generation.num_stores}")
#         print(f"  Items: {config.data.generation.num_items}")
        
#         # Print feature engineering settings
#         print("\n🔧 Feature Engineering:")
#         print(f"  Lag Windows: {config.features.lag_windows}")
#         print(f"  Rolling Windows: {config.features.rolling_windows}")
#         print(f"  Feature Selection: {config.features.use_feature_selection}")
        
#         # Print model settings
#         print("\n🤖 Model Settings:")
#         print(f"  N Estimators: {config.model.xgboost.n_estimators}")
#         print(f"  Max Depth: {config.model.xgboost.max_depth}")
#         print(f"  Learning Rate: {config.model.xgboost.learning_rate}")
        
#         print("\n" + "=" * 70)
#         print("✅ All configuration tests passed!")
#         print("=" * 70)
        
#         # Try loading secrets (won't fail if not present)
#         print("\n🔐 Loading secrets...")
#         secrets = load_secrets()
#         if secrets:
#             print(f"✅ Loaded {len(secrets)} secret categories")
#         else:
#             print("⚠️  No secrets file found (OK for development)")
        
#     except Exception as e:
#         print("\n" + "=" * 70)
#         print("❌ Configuration test failed!")
#         print("=" * 70)
#         print(f"\nError: {e}")
#         import traceback
#         traceback.print_exc()

"""
Configuration Management Module

This module handles loading and validating configuration files using Pydantic.
Pydantic provides automatic validation, type checking, and clear error messages.
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================ 
# Configuration Models - Using Pydantic for validation 
# ============================================================================ 

class ProjectConfig(BaseModel):
    """Project metadata configuration"""
    name: str
    version: str
    description: str


class DataGenerationConfig(BaseModel):
    """Data generation parameters"""
    start_date: str
    end_date: str
    num_stores: int = Field(gt=0, description="Must be positive")
    num_items: int = Field(gt=0, description="Must be positive")
    seasonality_strength: float = Field(ge=0, le=1)
    trend_strength: float = Field(ge=0, le=1)
    noise_level: float = Field(ge=0, le=1)
    anomaly_percentage: float = Field(ge=0, le=1)

    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        """Ensure dates are in correct format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got {v}")


class DataConfig(BaseModel):
    """Data paths and generation configuration"""
    raw_dir: str
    raw_sales: str
    raw_calendar: str
    processed_dir: str
    processed_data: str
    validation_dir: str
    generation: DataGenerationConfig

    @validator("raw_dir", "processed_dir", "validation_dir")
    def strip_trailing_slash(cls, v):
        """Ensure directory paths don't have trailing slashes"""
        return v.rstrip("/")


class FeatureConfig(BaseModel):
    """Feature engineering configuration"""
    lag_windows: List[int]
    rolling_windows: List[int]
    rolling_stats: List[str]
    extract_date_features: bool
    cyclical_encoding: bool
    use_feature_selection: bool
    selection_method: Literal["mutual_info", "correlation", "model_based"]
    max_features: int = Field(gt=0)
    train_size: float = Field(gt=0, lt=1)
    val_size: float = Field(gt=0, lt=1)
    test_size: float = Field(gt=0, lt=1)

    @validator("train_size", "val_size", "test_size")
    def validate_split_range(cls, v):
        """Ensure split fractions are between 0 and 1"""
        if not 0 < v < 1:
            raise ValueError("Split sizes must be between 0 and 1")
        return v


class XGBoostConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0, le=1)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    min_child_weight: int = Field(ge=0)
    gamma: float = Field(ge=0)
    objective: str
    random_state: int
    n_jobs: int


class IsolationForestConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_samples: int = Field(gt=0)
    contamination: float = Field(gt=0, lt=0.5)
    random_state: int
    n_jobs: int


class TuningConfig(BaseModel):
    enabled: bool
    n_trials: int = Field(gt=0)
    timeout: int = Field(gt=0)


class ModelConfig(BaseModel):
    forecasting_model: Literal["xgboost", "lightgbm", "random_forest"]
    anomaly_model: Literal["isolation_forest", "autoencoder"]
    xgboost: XGBoostConfig
    isolation_forest: IsolationForestConfig
    tuning: TuningConfig


class MLflowRegistryConfig(BaseModel):
    staging_alias: str
    production_alias: str


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    artifact_location: str
    backend_store_uri: str
    registry: MLflowRegistryConfig


class RateLimitConfig(BaseModel):
    enabled: bool
    requests_per_minute: int = Field(gt=0)


class CORSConfig(BaseModel):
    enabled: bool
    allow_origins: List[str]
    allow_methods: List[str]
    allow_headers: List[str]


class APIConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    reload: bool
    workers: int = Field(gt=0)
    timeout: int = Field(gt=0)
    rate_limit: RateLimitConfig
    cors: CORSConfig


class PrometheusConfig(BaseModel):
    enabled: bool
    port: int = Field(gt=0, lt=65536)


class ThresholdsConfig(BaseModel):
    max_latency_ms: int = Field(gt=0)
    max_error_rate: float = Field(ge=0, le=1)
    min_accuracy: float = Field(ge=0, le=1)


class DriftConfig(BaseModel):
    enabled: bool
    check_interval_hours: int = Field(gt=0)
    drift_threshold: float = Field(ge=0, le=1)
    reference_window_days: int = Field(gt=0)


class AlertsConfig(BaseModel):
    enabled: bool
    email_enabled: bool
    slack_enabled: bool


class MonitoringConfig(BaseModel):
    prometheus: PrometheusConfig
    collect_predictions: bool
    collect_latency: bool
    collect_errors: bool
    thresholds: ThresholdsConfig
    drift: DriftConfig
    alerts: AlertsConfig


class TimeBasedRetrainingConfig(BaseModel):
    enabled: bool
    interval_days: int = Field(gt=0)


class PerformanceBasedRetrainingConfig(BaseModel):
    enabled: bool
    rmse_threshold_increase: float = Field(gt=0)


class DriftBasedRetrainingConfig(BaseModel):
    enabled: bool
    num_drifted_features_threshold: int = Field(gt=0)


class DataVolumeRetrainingConfig(BaseModel):
    enabled: bool
    min_new_samples: int = Field(gt=0)


class RetrainingConfig(BaseModel):
    time_based: TimeBasedRetrainingConfig
    performance_based: PerformanceBasedRetrainingConfig
    drift_based: DriftBasedRetrainingConfig
    data_volume: DataVolumeRetrainingConfig


class SQLiteConfig(BaseModel):
    path: str


class PostgreSQLConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    database: str
    username: str


class DatabaseConfig(BaseModel):
    type: Literal["sqlite", "postgresql", "mysql"]
    sqlite: SQLiteConfig
    postgresql: PostgreSQLConfig


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: Literal["json", "text"]
    log_dir: str
    log_file: str
    rotation: str
    retention: str


class TestingConfig(BaseModel):
    test_data_size: int = Field(gt=0)
    random_seed: int


class DockerConfig(BaseModel):
    image_name: str
    image_tag: str


class KubernetesConfig(BaseModel):
    namespace: str
    replicas: int = Field(gt=0)
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str


class DeploymentConfig(BaseModel):
    environment: Literal["development", "staging", "production"]
    docker: DockerConfig
    kubernetes: KubernetesConfig


# ============================================================================ 
# Main Configuration Class 
# ============================================================================ 

class Config(BaseModel):
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    mlflow: MLflowConfig
    api: APIConfig
    monitoring: MonitoringConfig
    retraining: RetrainingConfig
    database: DatabaseConfig
    logging: LoggingConfig
    testing: TestingConfig
    deployment: DeploymentConfig

    class Config:
        validate_assignment = True
        extra = "forbid"


# ============================================================================ 
# Configuration Loading Functions 
# ============================================================================ 

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config").exists():
            return parent.parent
    raise RuntimeError("Could not find project root directory")


def load_yaml_file(file_path: Path) -> Dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    print(f"Loading configuration from: {config_path}")
    config_dict = load_yaml_file(config_path)
    config = Config(**config_dict)
    print("✅ Configuration loaded and validated successfully!")
    return config


def load_secrets(secrets_path: Optional[str] = None) -> Dict:
    if secrets_path is None:
        root = get_project_root()
        secrets_path = root / "config" / "secrets.yaml"
    else:
        secrets_path = Path(secrets_path)

    if not secrets_path.exists():
        print(f"⚠️  Secrets file not found: {secrets_path}")
        return {}

    print(f"Loading secrets from: {secrets_path}")
    secrets = load_yaml_file(secrets_path)
    print("✅ Secrets loaded successfully!")
    return secrets


def create_directories_from_config(config: Config) -> None:
    root = get_project_root()
    directories = [
        config.data.raw_dir,
        config.data.processed_dir,
        config.data.validation_dir,
        config.logging.log_dir,
        "models/registry",
        "models/staging",
        "models/production",
    ]
    for directory in directories:
        dir_path = root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ All required directories created!")


def get_config() -> Config:
    if not hasattr(get_config, "_config"):
        get_config._config = load_config()
        create_directories_from_config(get_config._config)
    return get_config._config


def get_secrets() -> Dict:
    if not hasattr(get_secrets, "_secrets"):
        get_secrets._secrets = load_secrets()
    return get_secrets._secrets


# ============================================================================ 
# Testing and Validation 
# ============================================================================ 

if __name__ == "__main__":
    print("=" * 70)
    print("Configuration Loading Test")
    print("=" * 70)

    try:
        config = load_config()

        print("\n📋 Key Configuration Settings:")
        print(f"  Project: {config.project.name} v{config.project.version}")
        print(f"  Environment: {config.deployment.environment}")
        print(f"  Model: {config.model.forecasting_model}")
        print(f"  API Port: {config.api.port}")
        print(f"  MLflow: {config.mlflow.tracking_uri}")

        print("\n📊 Data Generation:")
        print(f"  Date Range: {config.data.generation.start_date} to {config.data.generation.end_date}")
        print(f"  Stores: {config.data.generation.num_stores}")
        print(f"  Items: {config.data.generation.num_items}")

        print("\n🔧 Feature Engineering:")
        print(f"  Lag Windows: {config.features.lag_windows}")
        print(f"  Rolling Windows: {config.features.rolling_windows}")
        print(f"  Feature Selection: {config.features.use_feature_selection}")

        print("\n🤖 Model Settings:")
        print(f"  N Estimators: {config.model.xgboost.n_estimators}")
        print(f"  Max Depth: {config.model.xgboost.max_depth}")
        print(f"  Learning Rate: {config.model.xgboost.learning_rate}")

        print("\n" + "=" * 70)
        print("✅ All configuration tests passed!")
        print("=" * 70)

        print("\n🔐 Loading secrets...")
        secrets = load_secrets()
        if secrets:
            print(f"✅ Loaded {len(secrets)} secret categories")
        else:
            print("⚠️  No secrets file found (OK for development)")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Configuration test failed!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
