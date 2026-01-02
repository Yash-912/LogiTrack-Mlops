"""
Synthetic Data Generation Module

Generates realistic supply chain data with:
- Seasonal patterns (higher sales in certain months)
- Weekly patterns (weekday vs weekend differences)
- Trend (gradual increase/decrease over time)
- Noise (random variation)
- Anomalies (unusual spikes or drops)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config


def generate_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Generate a continuous date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DatetimeIndex with daily frequency
    """
    return pd.date_range(start=start_date, end=end_date, freq='D')


def add_seasonal_pattern(dates: pd.DatetimeIndex, strength: float = 0.3) -> np.ndarray:
    """
    Add seasonal pattern (yearly cycle).
    Sales are typically higher in Q4 (holidays) and lower in Q1.
    
    Args:
        dates: Array of dates
        strength: Strength of seasonality (0-1)
        
    Returns:
        Seasonal multiplier array
    """
    # Convert to day of year (1-365)
    day_of_year = dates.dayofyear.to_numpy()
    
    # Use sine wave with peak around day 350 (late December)
    # Phase shift to put peak at end of year
    seasonal = 1 + strength * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    return seasonal


def add_weekly_pattern(dates: pd.DatetimeIndex, strength: float = 0.2) -> np.ndarray:
    """
    Add weekly pattern.
    Sales are typically higher on weekends (Saturday=5, Sunday=6).
    
    Args:
        dates: Array of dates
        strength: Strength of weekly pattern (0-1)
        
    Returns:
        Weekly multiplier array
    """
    day_of_week = dates.dayofweek.to_numpy()
    
    # Create pattern: lower Mon-Thu, higher Fri-Sun
    weekly_multiplier = np.ones(len(dates))
    weekly_multiplier[day_of_week <= 3] = 1 - strength  # Mon-Thu
    weekly_multiplier[day_of_week >= 4] = 1 + strength  # Fri-Sun
    
    return weekly_multiplier


def add_trend(dates: pd.DatetimeIndex, strength: float = 0.2) -> np.ndarray:
    """
    Add linear trend over time.
    
    Args:
        dates: Array of dates
        strength: Strength of trend (0-1)
        
    Returns:
        Trend multiplier array
    """
    # Normalize time from 0 to 1
    time_normalized = np.arange(len(dates)) / len(dates)
    
    # Linear trend
    trend = 1 + strength * time_normalized
    
    return trend


def add_noise(size: int, noise_level: float = 0.1) -> np.ndarray:
    """
    Add random noise.
    
    Args:
        size: Number of samples
        noise_level: Standard deviation of noise (0-1)
        
    Returns:
        Noise multiplier array
    """
    return 1 + np.random.normal(0, noise_level, size)


    def inject_anomalies(sales: np.ndarray, anomaly_percentage: float = 0.05) -> tuple:
        """
        Inject random anomalies (spikes and drops).
        
        Args:
            sales: Original sales array
            anomaly_percentage: Percentage of data points to make anomalous
            
        Returns:
            Tuple of (sales_with_anomalies, anomaly_indices)
        """
        n_anomalies = int(len(sales) * anomaly_percentage)
        anomaly_indices = np.random.choice(len(sales), n_anomalies, replace=False)
        
        sales_with_anomalies = sales.copy()
        
        for idx in anomaly_indices:
            # 50% chance of spike, 50% chance of drop
            if np.random.random() > 0.5:
                # Spike: 2-5x normal value
                multiplier = np.random.uniform(2, 5)
            else:
                # Drop: 0.1-0.5x normal value
                multiplier = np.random.uniform(0.1, 0.5)
            
            sales_with_anomalies[idx] *= multiplier
        
        return sales_with_anomalies, anomaly_indices


def generate_sales_data() -> pd.DataFrame:
    """
    Generate synthetic sales data with realistic patterns.
    
    Returns:
        DataFrame with columns: date, store_id, item_id, sales, is_anomaly
    """
    config = get_config()
    gen_config = config.data.generation
    
    print("📊 Generating sales data...")
    print(f"   Date range: {gen_config.start_date} to {gen_config.end_date}")
    print(f"   Stores: {gen_config.num_stores}")
    print(f"   Items: {gen_config.num_items}")
    
    # Generate date range
    dates = generate_date_range(gen_config.start_date, gen_config.end_date)
    n_days = len(dates)
    
    # Create combinations of date, store, item
    data_list = []
    
    for store_id in range(1, gen_config.num_stores + 1):
        for item_id in range(1, gen_config.num_items + 1):
            # Base sales amount (different for each store-item combination)
            base_sales = np.random.uniform(50, 200)
            
            # Add patterns
            seasonal = add_seasonal_pattern(dates, gen_config.seasonality_strength)
            weekly = add_weekly_pattern(dates, 0.2)
            trend = add_trend(dates, gen_config.trend_strength)
            noise = add_noise(n_days, gen_config.noise_level)
            
            # Combine all patterns
            sales = base_sales * seasonal * weekly * trend * noise
            
            # Inject anomalies
            sales_with_anomalies, anomaly_indices = inject_anomalies(
                sales, 
                gen_config.anomaly_percentage
            )
            
            # Create is_anomaly flag
            is_anomaly = np.zeros(n_days, dtype=bool)
            is_anomaly[anomaly_indices] = True
            
            # Create DataFrame for this store-item combination
            df_temp = pd.DataFrame({
                'date': dates,
                'store_id': store_id,
                'item_id': item_id,
                'sales': sales_with_anomalies,
                'is_anomaly': is_anomaly
            })
            
            data_list.append(df_temp)
    
    # Combine all data
    df = pd.concat(data_list, ignore_index=True)
    
    # Round sales to 2 decimal places
    df['sales'] = df['sales'].round(2)
    
    # Sort by date, store, item
    df = df.sort_values(['date', 'store_id', 'item_id']).reset_index(drop=True)
    
    print(f"✅ Generated {len(df):,} sales records")
    print(f"   Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.2f}%)")
    
    return df


def generate_calendar_data() -> pd.DataFrame:
    """
    Generate calendar features for each date.
    
    Returns:
        DataFrame with date features and holiday information
    """
    config = get_config()
    gen_config = config.data.generation
    
    print("\n📅 Generating calendar data...")
    
    # Generate date range
    dates = generate_date_range(gen_config.start_date, gen_config.end_date)
    
    # Create DataFrame
    df = pd.DataFrame({'date': dates})
    
    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Boolean features
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_quarter_start'] = df['date'].dt.is_quarter_start
    df['is_quarter_end'] = df['date'].dt.is_quarter_end
    
    # Add simple holiday flags (US holidays for example)
    df['is_holiday'] = False
    
    # New Year's Day
    df.loc[(df['month'] == 1) & (df['day'] == 1), 'is_holiday'] = True
    
    # Independence Day
    df.loc[(df['month'] == 7) & (df['day'] == 4), 'is_holiday'] = True
    
    # Christmas
    df.loc[(df['month'] == 12) & (df['day'] == 25), 'is_holiday'] = True
    
    # Thanksgiving (4th Thursday of November - approximation)
    df.loc[(df['month'] == 11) & (df['day'].between(22, 28)) & 
           (df['day_of_week'] == 3), 'is_holiday'] = True
    
    print(f"✅ Generated calendar for {len(df):,} days")
    print(f"   Holidays: {df['is_holiday'].sum()}")
    print(f"   Weekends: {df['is_weekend'].sum()}")
    
    return df


def save_raw_data(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> None:
    """
    Save raw data to CSV files.
    
    Args:
        sales_df: Sales data
        calendar_df: Calendar data
    """
    config = get_config()
    
    # Create directory if it doesn't exist
    raw_dir = Path(config.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    sales_path = Path(config.data.raw_sales)
    calendar_path = Path(config.data.raw_calendar)
    
    print(f"\n💾 Saving data...")
    print(f"   Sales: {sales_path}")
    print(f"   Calendar: {calendar_path}")
    
    sales_df.to_csv(sales_path, index=False)
    calendar_df.to_csv(calendar_path, index=False)
    
    print("✅ Data saved successfully!")
    
    # Print file sizes
    sales_size = sales_path.stat().st_size / (1024 * 1024)  # MB
    calendar_size = calendar_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Sales file size: {sales_size:.2f} MB")
    print(f"   Calendar file size: {calendar_size:.2f} MB")


def generate_data_version_metadata(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> dict:
    """
    Generate metadata about the generated data.
    
    Args:
        sales_df: Sales data
        calendar_df: Calendar data
        
    Returns:
        Dictionary with metadata
    """
    config = get_config()
    
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'data_version': '1.0.0',
        'config_version': config.project.version,
        'sales_data': {
            'num_records': len(sales_df),
            'num_stores': sales_df['store_id'].nunique(),
            'num_items': sales_df['item_id'].nunique(),
            'date_range': {
                'start': sales_df['date'].min().strftime('%Y-%m-%d'),
                'end': sales_df['date'].max().strftime('%Y-%m-%d')
            },
            'sales_stats': {
                'mean': float(sales_df['sales'].mean()),
                'std': float(sales_df['sales'].std()),
                'min': float(sales_df['sales'].min()),
                'max': float(sales_df['sales'].max())
            },
            'anomalies': {
                'count': int(sales_df['is_anomaly'].sum()),
                'percentage': float(sales_df['is_anomaly'].mean() * 100)
            }
        },
        'calendar_data': {
            'num_records': len(calendar_df),
            'num_holidays': int(calendar_df['is_holiday'].sum()),
            'num_weekends': int(calendar_df['is_weekend'].sum())
        },
        'generation_parameters': {
            'start_date': config.data.generation.start_date,
            'end_date': config.data.generation.end_date,
            'num_stores': config.data.generation.num_stores,
            'num_items': config.data.generation.num_items,
            'seasonality_strength': config.data.generation.seasonality_strength,
            'trend_strength': config.data.generation.trend_strength,
            'noise_level': config.data.generation.noise_level,
            'anomaly_percentage': config.data.generation.anomaly_percentage
        }
    }
    
    return metadata


def save_metadata(metadata: dict) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
    """
    config = get_config()
    
    metadata_path = Path(config.data.raw_dir) / 'data_metadata.json'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    
    print(f"\n📄 Metadata saved: {metadata_path}")


def print_data_summary(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> None:
    """
    Print a summary of the generated data.
    
    Args:
        sales_df: Sales data
        calendar_df: Calendar data
    """
    print("\n" + "=" * 70)
    print("📊 DATA GENERATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 Sales Data:")
    print(f"   Total records: {len(sales_df):,}")
    print(f"   Stores: {sales_df['store_id'].nunique()}")
    print(f"   Items: {sales_df['item_id'].nunique()}")
    print(f"   Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"   Sales range: ${sales_df['sales'].min():.2f} to ${sales_df['sales'].max():.2f}")
    print(f"   Average sales: ${sales_df['sales'].mean():.2f}")
    print(f"   Anomalies: {sales_df['is_anomaly'].sum():,} ({sales_df['is_anomaly'].mean()*100:.2f}%)")
    
    print(f"\n📅 Calendar Data:")
    print(f"   Total days: {len(calendar_df):,}")
    print(f"   Holidays: {calendar_df['is_holiday'].sum()}")
    print(f"   Weekends: {calendar_df['is_weekend'].sum()}")
    
    print(f"\n🔍 Sample Sales Data (first 10 rows):")
    print(sales_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)


def main():
    """Main execution function"""
    print("=" * 70)
    print("🚀 SYNTHETIC DATA GENERATION")
    print("=" * 70)
    
    # Generate data
    sales_df = generate_sales_data()
    calendar_df = generate_calendar_data()
    
    # Save raw data
    save_raw_data(sales_df, calendar_df)
    
    # Generate and save metadata
    metadata = generate_data_version_metadata(sales_df, calendar_df)
    save_metadata(metadata)
    
    # Print summary
    print_data_summary(sales_df, calendar_df)
    
    print("\n✅ Data generation complete!")
    print("\n📝 Next steps:")
    print("   1. Track data with DVC: dvc add data/raw/sales_data.csv")
    print("   2. Track data with DVC: dvc add data/raw/calendar_data.csv")
    print("   3. Commit DVC files: git add data/raw/*.dvc")
    print("   4. Run data validation")
    
    return sales_df, calendar_df


if __name__ == "__main__":
    main()