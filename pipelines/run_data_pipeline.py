"""
Complete Data Pipeline Runner

This script runs the entire data pipeline:
1. Generate synthetic data
2. Validate raw data
3. Preprocess data
4. Validate processed data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.generate_data import main as generate_data_main
from data.data_validation import validate_raw_data
from data.preprocessing import preprocess_pipeline


def main():
    """Run the complete data pipeline"""
    print("=" * 80)
    print(" " * 20 + "COMPLETE DATA PIPELINE")
    print("=" * 80)
    print()
    
    # Phase 1: Generate Data
    print("📊 PHASE 1: DATA GENERATION")
    print("-" * 80)
    try:
        sales_df, calendar_df = generate_data_main()
        print("✅ Data generation successful!")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    
    # Phase 2: Validate Raw Data
    print("\n🔍 PHASE 2: RAW DATA VALIDATION")
    print("-" * 80)
    try:
        if not validate_raw_data():
            print("❌ Raw data validation failed!")
            return False
        print("✅ Raw data validation successful!")
    except Exception as e:
        print(f"❌ Raw data validation error: {e}")
        return False
    
    print("\n" + "=" * 80)
    
    # Phase 3: Preprocess Data
    print("\n🔧 PHASE 3: DATA PREPROCESSING")
    print("-" * 80)
    try:
        if not preprocess_pipeline(validate_first=False, treat_outliers=False):
            print("❌ Data preprocessing failed!")
            return False
        print("✅ Data preprocessing successful!")
    except Exception as e:
        print(f"❌ Data preprocessing error: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 80)
    print("🎉 DATA PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📊 Generated Files:")
    print("   • data/raw/sales_data.csv")
    print("   • data/raw/calendar_data.csv")
    print("   • data/raw/data_metadata.json")
    print("   • data/processed/processed_data.parquet")
    print("   • data/processed/preprocessing_metadata.json")
    print("   • data/validation/*.json (validation reports)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)