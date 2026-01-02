"""
Quick test script to verify configuration setup.
Run this after creating all config files.

Usage:
    python test_config.py
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config.config import load_config, load_secrets, get_config


def test_config_loading():
    """Test that configuration loads correctly"""
    print("🧪 Testing configuration loading...")
    
    try:
        config = load_config()
        print("✅ Config loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False


def test_config_access():
    """Test accessing various config values"""
    print("\n🧪 Testing configuration access...")
    
    try:
        config = get_config()
        
        # Test accessing nested values
        assert config.project.name == "supply-chain-mlops"
        assert config.model.forecasting_model in ["xgboost", "lightgbm", "random_forest"]
        assert config.api.port > 0
        assert len(config.features.lag_windows) > 0
        
        print("✅ Configuration access works!")
        return True
    except Exception as e:
        print(f"❌ Configuration access failed: {e}")
        return False


def test_secrets_loading():
    """Test secrets loading (optional)"""
    print("\n🧪 Testing secrets loading...")
    
    try:
        secrets = load_secrets()
        if secrets:
            print(f"✅ Secrets loaded: {len(secrets)} categories")
        else:
            print("⚠️  No secrets file (OK for development)")
        return True
    except Exception as e:
        print(f"❌ Secrets loading failed: {e}")
        return False


def print_config_summary():
    """Print a summary of the configuration"""
    print("\n📋 Configuration Summary:")
    print("=" * 60)
    
    config = get_config()
    
    print(f"\n🏷️  PROJECT")
    print(f"   Name: {config.project.name}")
    print(f"   Version: {config.project.version}")
    print(f"   Environment: {config.deployment.environment}")
    
    print(f"\n📊 DATA")
    print(f"   Stores: {config.data.generation.num_stores}")
    print(f"   Items: {config.data.generation.num_items}")
    print(f"   Date Range: {config.data.generation.start_date} to {config.data.generation.end_date}")
    
    print(f"\n🔧 FEATURES")
    print(f"   Lag Windows: {config.features.lag_windows}")
    print(f"   Rolling Windows: {config.features.rolling_windows}")
    print(f"   Train/Val/Test Split: {config.features.train_size}/{config.features.val_size}/{config.features.test_size}")
    
    print(f"\n🤖 MODEL")
    print(f"   Forecasting Model: {config.model.forecasting_model}")
    print(f"   Anomaly Model: {config.model.anomaly_model}")
    print(f"   N Estimators: {config.model.xgboost.n_estimators}")
    print(f"   Max Depth: {config.model.xgboost.max_depth}")
    
    print(f"\n🌐 API")
    print(f"   Host: {config.api.host}")
    print(f"   Port: {config.api.port}")
    print(f"   Workers: {config.api.workers}")
    
    print(f"\n📈 MLFLOW")
    print(f"   Tracking URI: {config.mlflow.tracking_uri}")
    print(f"   Experiment: {config.mlflow.experiment_name}")
    
    print("\n" + "=" * 60)


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 CONFIGURATION VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_config_access,
        test_secrets_loading,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Print summary
    print_config_summary()
    
    # Final result
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n🎉 Configuration is ready to use!")
        print("\n📝 Next steps:")
        print("   1. Review the configuration summary above")
        print("   2. Modify config/config.yaml if needed")
        print("   3. Move to Phase II: Data Pipeline")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n🔧 Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())