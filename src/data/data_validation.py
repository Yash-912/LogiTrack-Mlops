"""
Data Validation Module using Great Expectations

This module ensures data quality by defining and checking expectations:
- Schema validation (correct columns and types)
- Value range validation (e.g., sales >= 0)
- Completeness validation (no missing values in critical columns)
- Uniqueness validation
- Statistical validation
"""

import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context import FileDataContext
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


class DataValidator:
    """
    Data validation class using Great Expectations.
    Validates both raw and processed data.
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.config = get_config()
        self.validation_dir = Path(self.config.data.validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_sales_data(self, df: pd.DataFrame, data_type: str = "raw") -> dict:
        """
        Validate sales data against expectations.
        
        Args:
            df: Sales DataFrame
            data_type: Type of data ("raw" or "processed")
            
        Returns:
            Validation results dictionary
        """
        print(f"\n🔍 Validating {data_type} sales data...")
        print(f"   Shape: {df.shape}")
        
        results = {
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'expectations': []
        }
        
        # Define expectations
        expectations = [
            # Column existence
            ('date column exists', 
             lambda: 'date' in df.columns),
            
            ('store_id column exists', 
             lambda: 'store_id' in df.columns),
            
            ('item_id column exists', 
             lambda: 'item_id' in df.columns),
            
            ('sales column exists', 
             lambda: 'sales' in df.columns),
            
            # No null values in critical columns
            ('date has no nulls', 
             lambda: df['date'].notna().all()),
            
            ('store_id has no nulls', 
             lambda: df['store_id'].notna().all()),
            
            ('item_id has no nulls', 
             lambda: df['item_id'].notna().all()),
            
            ('sales has no nulls', 
             lambda: df['sales'].notna().all()),
            
            # Data types
            ('sales is numeric', 
             lambda: pd.api.types.is_numeric_dtype(df['sales'])),
            
            ('store_id is numeric', 
             lambda: pd.api.types.is_numeric_dtype(df['store_id'])),
            
            ('item_id is numeric', 
             lambda: pd.api.types.is_numeric_dtype(df['item_id'])),
            
            # Value ranges
            ('sales >= 0', 
             lambda: (df['sales'] >= 0).all()),
            
            ('store_id > 0', 
             lambda: (df['store_id'] > 0).all()),
            
            ('item_id > 0', 
             lambda: (df['item_id'] > 0).all()),
            
            # Statistical checks
            ('sales has reasonable mean (10-500)', 
             lambda: 10 <= df['sales'].mean() <= 500),
            
            ('sales std > 0', 
             lambda: df['sales'].std() > 0),
            
            # Data quality
            ('no duplicate rows', 
             lambda: df.duplicated().sum() == 0),
            
            ('dates are in order', 
             lambda: df['date'].is_monotonic_increasing or 
                     df.groupby(['store_id', 'item_id'])['date'].is_monotonic_increasing.all()),
        ]
        
        # Run expectations
        for expectation_name, check_func in expectations:
            results['total_checks'] += 1
            try:
                passed = check_func()
                if passed:
                    results['passed_checks'] += 1
                    status = '✅'
                else:
                    results['failed_checks'] += 1
                    status = '❌'
                
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': bool(passed),
                    'status': status
                })
                
                print(f"   {status} {expectation_name}")
                
            except Exception as e:
                results['failed_checks'] += 1
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': False,
                    'status': '❌',
                    'error': str(e)
                })
                print(f"   ❌ {expectation_name} - Error: {e}")
        
        # Calculate pass rate
        results['pass_rate'] = results['passed_checks'] / results['total_checks']
        
        return results
    
    def validate_calendar_data(self, df: pd.DataFrame) -> dict:
        """
        Validate calendar data.
        
        Args:
            df: Calendar DataFrame
            
        Returns:
            Validation results dictionary
        """
        print(f"\n🔍 Validating calendar data...")
        print(f"   Shape: {df.shape}")
        
        results = {
            'data_type': 'calendar',
            'timestamp': datetime.now().isoformat(),
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'expectations': []
        }
        
        expectations = [
            # Column existence
            ('date column exists', 
             lambda: 'date' in df.columns),
            
            ('day_of_week column exists', 
             lambda: 'day_of_week' in df.columns),
            
            ('is_holiday column exists', 
             lambda: 'is_holiday' in df.columns),
            
            ('is_weekend column exists', 
             lambda: 'is_weekend' in df.columns),
            
            # No null values
            ('date has no nulls', 
             lambda: df['date'].notna().all()),
            
            # Value ranges
            ('day_of_week is 0-6', 
             lambda: df['day_of_week'].between(0, 6).all()),
            
            ('is_holiday is boolean', 
             lambda: df['is_holiday'].dtype == bool or df['is_holiday'].isin([0, 1, True, False]).all()),
            
            ('is_weekend is boolean', 
             lambda: df['is_weekend'].dtype == bool or df['is_weekend'].isin([0, 1, True, False]).all()),
            
            # Uniqueness
            ('dates are unique', 
             lambda: df['date'].nunique() == len(df)),
            
            # Consistency
            ('weekend matches day_of_week', 
             lambda: (df['is_weekend'] == df['day_of_week'].isin([5, 6])).all()),
        ]
        
        # Run expectations
        for expectation_name, check_func in expectations:
            results['total_checks'] += 1
            try:
                passed = check_func()
                if passed:
                    results['passed_checks'] += 1
                    status = '✅'
                else:
                    results['failed_checks'] += 1
                    status = '❌'
                
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': bool(passed),
                    'status': status
                })
                
                print(f"   {status} {expectation_name}")
                
            except Exception as e:
                results['failed_checks'] += 1
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': False,
                    'status': '❌',
                    'error': str(e)
                })
                print(f"   ❌ {expectation_name} - Error: {e}")
        
        results['pass_rate'] = results['passed_checks'] / results['total_checks']
        
        return results
    
    def validate_processed_data(self, df: pd.DataFrame) -> dict:
        """
        Validate processed/merged data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Validation results dictionary
        """
        print(f"\n🔍 Validating processed data...")
        print(f"   Shape: {df.shape}")
        
        results = {
            'data_type': 'processed',
            'timestamp': datetime.now().isoformat(),
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'expectations': []
        }
        
        expectations = [
            # Required columns
            ('has date column', 
             lambda: 'date' in df.columns),
            
            ('has sales column', 
             lambda: 'sales' in df.columns),
            
            ('has store_id column', 
             lambda: 'store_id' in df.columns),
            
            ('has item_id column', 
             lambda: 'item_id' in df.columns),
            
            # No critical nulls after processing
            ('no nulls in sales', 
             lambda: df['sales'].notna().all()),
            
            ('no nulls in date features', 
             lambda: df[['year', 'month', 'day']].notna().all().all() if 'year' in df.columns else True),
            
            # Data quality
            ('no infinite values in sales', 
             lambda: ~df['sales'].isin([float('inf'), float('-inf')]).any()),
            
            ('sales variance > 0', 
             lambda: df['sales'].std() > 0),
        ]
        
        # Run expectations
        for expectation_name, check_func in expectations:
            results['total_checks'] += 1
            try:
                passed = check_func()
                if passed:
                    results['passed_checks'] += 1
                    status = '✅'
                else:
                    results['failed_checks'] += 1
                    status = '❌'
                
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': bool(passed),
                    'status': status
                })
                
                print(f"   {status} {expectation_name}")
                
            except Exception as e:
                results['failed_checks'] += 1
                results['expectations'].append({
                    'name': expectation_name,
                    'passed': False,
                    'status': '❌',
                    'error': str(e)
                })
                print(f"   ❌ {expectation_name} - Error: {e}")
        
        results['pass_rate'] = results['passed_checks'] / results['total_checks']
        
        return results
    
    def save_validation_report(self, results: dict, report_name: str = "validation_report") -> None:
        """
        Save validation results to JSON file.
        
        Args:
            results: Validation results dictionary
            report_name: Name for the report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.validation_dir / f"{report_name}_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Validation report saved: {report_path}")
    
    def print_validation_summary(self, results: dict) -> None:
        """
        Print a summary of validation results.
        
        Args:
            results: Validation results dictionary
        """
        print("\n" + "=" * 70)
        print(f"📊 VALIDATION SUMMARY - {results['data_type'].upper()}")
        print("=" * 70)
        print(f"Total Checks: {results['total_checks']}")
        print(f"✅ Passed: {results['passed_checks']}")
        print(f"❌ Failed: {results['failed_checks']}")
        print(f"Pass Rate: {results['pass_rate']*100:.1f}%")
        
        if results['failed_checks'] > 0:
            print("\n⚠️  Failed Expectations:")
            for exp in results['expectations']:
                if not exp['passed']:
                    print(f"   • {exp['name']}")
                    if 'error' in exp:
                        print(f"     Error: {exp['error']}")
        
        print("=" * 70)


def validate_raw_data() -> bool:
    """
    Validate raw data files.
    
    Returns:
        True if validation passes, False otherwise
    """
    config = get_config()
    validator = DataValidator()
    
    print("=" * 70)
    print("🔍 RAW DATA VALIDATION")
    print("=" * 70)
    
    # Load raw data
    print("\n📂 Loading raw data...")
    sales_df = pd.read_csv(config.data.raw_sales)
    calendar_df = pd.read_csv(config.data.raw_calendar)
    
    # Convert date columns
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    print(f"✅ Loaded sales data: {sales_df.shape}")
    print(f"✅ Loaded calendar data: {calendar_df.shape}")
    
    # Validate sales data
    sales_results = validator.validate_sales_data(sales_df, "raw")
    validator.print_validation_summary(sales_results)
    validator.save_validation_report(sales_results, "sales_raw_validation")
    
    # Validate calendar data
    calendar_results = validator.validate_calendar_data(calendar_df)
    validator.print_validation_summary(calendar_results)
    validator.save_validation_report(calendar_results, "calendar_validation")
    
    # Overall result
    all_passed = (sales_results['failed_checks'] == 0 and 
                  calendar_results['failed_checks'] == 0)
    
    if all_passed:
        print("\n✅ All validations passed! Data is ready for preprocessing.")
    else:
        print("\n⚠️  Some validations failed. Please review and fix issues.")
    
    return all_passed


if __name__ == "__main__":
    """Run validation on raw data"""
    success = validate_raw_data()
    sys.exit(0 if success else 1)