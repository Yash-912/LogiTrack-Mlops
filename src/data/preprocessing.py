"""
Data Preprocessing Module

Handles:
- Loading raw data
- Date conversion
- Merging datasets
- Handling missing values
- Outlier detection and treatment
- Saving processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config
from data.data_validation import DataValidator


class DataPreprocessor:
    """
    Data preprocessing pipeline for supply chain data.
    """
    
        def __init__(self):
            """Initialize preprocessor with configuration"""
            self.config = get_config()
            self.validator = DataValidator()
            self.preprocessing_stats = {
                'timestamp': datetime.now().isoformat(),
                'steps_completed': [],
                'statistics': {}
            }
        
    def load_raw_data(self) -> tuple:
        """
        Load raw sales and calendar data.,
        
        Returns:
            Tuple of (sales_df, calendar_df)
        """
        print("\n📂 Loading raw data...")
        
        # Load CSVs
        sales_df = pd.read_csv(self.config.data.raw_sales)
        calendar_df = pd.read_csv(self.config.data.raw_calendar)
        
        print(f"   ✅ Sales: {sales_df.shape}")
        print(f"   ✅ Calendar: {calendar_df.shape}")
        
        # Store initial stats
        self.preprocessing_stats['statistics']['raw_sales_rows'] = len(sales_df)
        self.preprocessing_stats['statistics']['raw_calendar_rows'] = len(calendar_df)
        
        return sales_df, calendar_df
    
    def convert_dates(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> tuple:
        """
        Convert date columns to datetime type.
        
        Args:
            sales_df: Sales DataFrame
            calendar_df: Calendar DataFrame
            
        Returns:
            Tuple of DataFrames with converted dates
        """
        print("\n📅 Converting date columns...")
        
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        print("   ✅ Dates converted to datetime")
        
        self.preprocessing_stats['steps_completed'].append('date_conversion')
        
        return sales_df, calendar_df
    
    def merge_datasets(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sales and calendar data on date.
        
        Args:
            sales_df: Sales DataFrame
            calendar_df: Calendar DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("\n🔗 Merging sales and calendar data...")
        
        # Merge on date
        merged_df = sales_df.merge(calendar_df, on='date', how='left')
        
        print(f"   ✅ Merged shape: {merged_df.shape}")
        print(f"   Columns: {list(merged_df.columns)}")
        
        # Check for missing values from merge
        merge_nulls = merged_df.isnull().sum()
        if merge_nulls.any():
            print(f"   ⚠️  Null values after merge:")
            print(merge_nulls[merge_nulls > 0])
        
        self.preprocessing_stats['statistics']['merged_rows'] = len(merged_df)
        self.preprocessing_stats['steps_completed'].append('merge_datasets')
        
        return merged_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Sales: Forward fill then backward fill (reasonable for time series)
        - Categorical: Fill with 'Unknown' or mode
        - Boolean: Fill with False (conservative approach)
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        print("\n🔧 Handling missing values...")
        
        initial_nulls = df.isnull().sum()
        if initial_nulls.sum() == 0:
            print("   ✅ No missing values found")
            return df
        
        print(f"   Missing values before imputation:")
        print(initial_nulls[initial_nulls > 0])
        
        # Handle sales (forward fill then backward fill)
        if 'sales' in df.columns and df['sales'].isnull().any():
            df['sales'] = df.groupby(['store_id', 'item_id'])['sales'].fillna(method='ffill')
            df['sales'] = df.groupby(['store_id', 'item_id'])['sales'].fillna(method='bfill')           
            print(f"   ✅ Imputed sales with forward/backward fill")
        
        # Handle boolean columns
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(False)
                print(f"   ✅ Filled {col} with False")
        
        # Handle numeric columns (if any remaining)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                print(f"   ✅ Filled {col} with median")
        
        # Check final nulls
        final_nulls = df.isnull().sum()
        if final_nulls.sum() > 0:
            print(f"   ⚠️  Remaining nulls:")
            print(final_nulls[final_nulls > 0])
        else:
            print(f"   ✅ All missing values handled")
        
    
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in sales data using IQR method.
        
        Args:
            df: DataFrame
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outlier flag added
        """
        print(f"\n🔍 Detecting outliers using {method.upper()} method...")
        
        if method == 'iqr':
            # IQR method (per store-item combination)
            df['is_outlier'] = False
            
            for (store, item), group in df.groupby(['store_id', 'item_id']):
                Q1 = group['sales'].quantile(0.25)
                Q3 = group['sales'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # 3*IQR for more conservative
                upper_bound = Q3 + 3 * IQR
                
                outliers = (group['sales'] < lower_bound) | (group['sales'] > upper_bound)
                df.loc[group.index[outliers], 'is_outlier'] = True
        
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(df['sales']))
            df['is_outlier'] = z_scores > 3  # 3 standard deviations
        
        outlier_count = df['is_outlier'].sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        print(f"   ✅ Detected {outlier_count:,} outliers ({outlier_pct:.2f}%)")
        
        self.preprocessing_stats['statistics']['outliers_detected'] = int(outlier_count)
        self.preprocessing_stats['statistics']['outlier_percentage'] = float(outlier_pct)
        self.preprocessing_stats['steps_completed'].append('detect_outliers')
        
        return df
    
    def treat_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Treat outliers (optional step).
        
        Args:
            df: DataFrame with outlier flag
            method: Treatment method ('cap', 'remove', 'none')
            
        Returns:
            DataFrame with treated outliers
        """
        if method == 'none' or 'is_outlier' not in df.columns:
            return df
        
        print(f"\n🔧 Treating outliers using '{method}' method...")
        
        outlier_count = df['is_outlier'].sum()
        
        if method == 'cap':
            # Cap outliers at percentiles (per store-item)
            for (store, item), group in df.groupby(['store_id', 'item_id']):
                lower = group['sales'].quantile(0.01)
                upper = group['sales'].quantile(0.99)
                
                mask = (df['store_id'] == store) & (df['item_id'] == item) & df['is_outlier']
                df.loc[mask & (df['sales'] < lower), 'sales'] = lower
                df.loc[mask & (df['sales'] > upper), 'sales'] = upper
            
            print(f"   ✅ Capped {outlier_count:,} outliers")
        
        elif method == 'remove':
            # Remove outliers (not recommended for time series)
            df = df[~df['is_outlier']]
            print(f"   ✅ Removed {outlier_count:,} outliers")
            print(f"   New shape: {df.shape}")
        
        self.preprocessing_stats['statistics']['outliers_treated'] = int(outlier_count)
        self.preprocessing_stats['statistics']['outlier_treatment_method'] = method
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic derived features for easier analysis.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with additional features
        """
        print("\n🔧 Adding basic features...")
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract additional date features if not present
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['date'].dt.day
        
        print(f"   ✅ Added basic date features")
        
        self.preprocessing_stats['steps_completed'].append('add_basic_features')
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Save processed data to parquet file.
        
        Args:
            df: Processed DataFrame
        """
        print("\n💾 Saving processed data...")
        
        # Create directory if needed
        processed_dir = Path(self.config.data.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet (more efficient than CSV)
        output_path = Path(self.config.data.processed_data)
        df.to_parquet(output_path, index=False)
        
        # Get file size
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"   ✅ Saved to: {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Shape: {df.shape}")
        
        self.preprocessing_stats['statistics']['final_rows'] = len(df)
        self.preprocessing_stats['statistics']['final_columns'] = len(df.columns)
        self.preprocessing_stats['statistics']['file_size_mb'] = float(file_size)
    
    def save_preprocessing_metadata(self) -> None:
        """Save preprocessing statistics and metadata"""
        metadata_path = Path(self.config.data.processed_dir) / 'preprocessing_metadata.json'
        
        with open(metadata_path, 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2)
        
        print(f"\n📄 Preprocessing metadata saved: {metadata_path}")
    
    def run(self, treat_outliers: bool = False) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            treat_outliers: Whether to treat outliers (default: False, keep for anomaly detection)
            
        Returns:
            Processed DataFrame
        """
        print("=" * 70)
        print("🚀 DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        sales_df, calendar_df = self.load_raw_data()
        
        # Step 2: Convert dates
        sales_df, calendar_df = self.convert_dates(sales_df, calendar_df)
        
        # Step 3: Merge datasets
        df = self.merge_datasets(sales_df, calendar_df)
        
        # Step 4: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 5: Detect outliers
        df = self.detect_outliers(df, method='iqr')
        
        # Step 6: Treat outliers (optional)
        if treat_outliers:
            df = self.treat_outliers(df, method='cap')
        else:
            print("\n⚠️  Outliers detected but not treated (useful for anomaly detection)")
        
        # Step 7: Add basic features
        df = self.add_basic_features(df)
        
        # Step 8: Save processed data
        self.save_processed_data(df)
        
        # Step 9: Save metadata
        self.save_preprocessing_metadata()
        
        # Print summary
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print preprocessing summary.
        
        Args:
            df: Processed DataFrame
        """
        print("\n" + "=" * 70)
        print("📊 PREPROCESSING SUMMARY")
        print("=" * 70)
        
        print(f"\n✅ Steps completed: {len(self.preprocessing_stats['steps_completed'])}")
        for step in self.preprocessing_stats['steps_completed']:
            print(f"   • {step}")
        
        print(f"\n📈 Statistics:")
        stats = self.preprocessing_stats['statistics']
        if 'raw_sales_rows' in stats:
            print(f"   Raw sales rows: {stats['raw_sales_rows']:,}")
        if 'final_rows' in stats:
            print(f"   Final rows: {stats['final_rows']:,}")
        if 'final_columns' in stats:
            print(f"   Final columns: {stats['final_columns']}")
        if 'nulls_before' in stats:
            print(f"   Nulls handled: {stats['nulls_before']:,}")
        if 'outliers_detected' in stats:
            print(f"   Outliers detected: {stats['outliers_detected']:,} ({stats['outlier_percentage']:.2f}%)")
        
        print(f"\n🔍 Data Quality:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {df.duplicated().sum()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        print(f"\n📋 Sample data (first 5 rows):")
        print(df.head().to_string())
        
        print("\n" + "=" * 70)


def preprocess_pipeline(validate_first: bool = True, treat_outliers: bool = False) -> bool:
    """
    Run the complete preprocessing pipeline with optional validation.
    
    Args:
        validate_first: Whether to validate raw data before processing
        treat_outliers: Whether to treat detected outliers
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Validate raw data (optional)
        if validate_first:
            print("🔍 Step 1: Validating raw data...")
            from data.data_validation import validate_raw_data
            
            if not validate_raw_data():
                print("❌ Raw data validation failed. Fix issues before preprocessing.")
                return False
        
        # Step 2: Run preprocessing
        print("\n🔧 Step 2: Running preprocessing pipeline...")
        preprocessor = DataPreprocessor()
        df = preprocessor.run(treat_outliers=treat_outliers)
        
        # Step 3: Validate processed data
        print("\n🔍 Step 3: Validating processed data...")
        validator = DataValidator()
        results = validator.validate_processed_data(df)
        validator.print_validation_summary(results)
        validator.save_validation_report(results, "processed_data_validation")
        
        if results['failed_checks'] > 0:
            print("⚠️  Some processed data validations failed. Review before proceeding.")
            return False
        
        print("\n✅ Preprocessing pipeline completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Review processed data in data/processed/")
        print("   2. Check validation reports in data/validation/")
        print("   3. Proceed to feature engineering")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Preprocessing pipeline failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run preprocessing pipeline"""
    success = preprocess_pipeline(validate_first=True, treat_outliers=False)
    sys.exit(0 if success else 1)