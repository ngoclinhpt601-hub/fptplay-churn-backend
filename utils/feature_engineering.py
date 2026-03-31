"""
Feature Engineering Module
Transforms raw customer data into ML-ready features
"""

import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Apply feature engineering to customer data
    
    Args:
        df: DataFrame with raw customer data
        
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    # ========== L1M (Last 1 Month) Features ==========
    df_eng['AVG_L1M_HOURS'] = df_eng['hours_m1']
    df_eng['MAX_L1M_HOURS'] = df_eng['hours_m1']
    df_eng['MIN_L1M_HOURS'] = df_eng['hours_m1']
    
    # ========== L3M (Last 3 Months) Features ==========
    df_eng['AVG_L3M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3']].mean(axis=1)
    df_eng['MAX_L3M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3']].max(axis=1)
    df_eng['MIN_L3M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3']].min(axis=1)
    df_eng['SUM_L3M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3']].sum(axis=1)
    df_eng['STDDEV_L3M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3']].std(axis=1)
    
    # Coefficient of Variation (CV)
    df_eng['CV_L3M_HOURS'] = df_eng['STDDEV_L3M_HOURS'] / (df_eng['AVG_L3M_HOURS'] + 1e-6)
    df_eng['CV_L3M_HOURS'] = df_eng['CV_L3M_HOURS'].fillna(0)
    
    # ========== L6M (Last 6 Months) Features ==========
    df_eng['AVG_L6M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3', 
                                       'hours_m4', 'hours_m5', 'hours_m6']].mean(axis=1)
    df_eng['MAX_L6M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3', 
                                       'hours_m4', 'hours_m5', 'hours_m6']].max(axis=1)
    df_eng['MIN_L6M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3', 
                                       'hours_m4', 'hours_m5', 'hours_m6']].min(axis=1)
    df_eng['SUM_L6M_HOURS'] = df_eng[['hours_m1', 'hours_m2', 'hours_m3', 
                                       'hours_m4', 'hours_m5', 'hours_m6']].sum(axis=1)
    
    # ========== Growth Rate Features ==========
    # L1M vs L3M
    df_eng['GROWTH_RATE_L1M_VS_L3M'] = (
        (df_eng['AVG_L1M_HOURS'] - df_eng['AVG_L3M_HOURS']) / (df_eng['AVG_L3M_HOURS'] + 1e-6)
    )
    df_eng['GROWTH_RATE_L1M_VS_L3M'] = df_eng['GROWTH_RATE_L1M_VS_L3M'].fillna(0)
    
    # L3M vs L6M
    df_eng['GROWTH_RATE_L3M_VS_L6M'] = (
        (df_eng['AVG_L3M_HOURS'] - df_eng['AVG_L6M_HOURS']) / (df_eng['AVG_L6M_HOURS'] + 1e-6)
    )
    df_eng['GROWTH_RATE_L3M_VS_L6M'] = df_eng['GROWTH_RATE_L3M_VS_L6M'].fillna(0)
    
    # ========== Trend Analysis ==========
    # Calculate trend slope (linear regression)
    def calculate_trend_slope(row):
        months = np.arange(6, 0, -1)  # [6, 5, 4, 3, 2, 1]
        hours = [row['hours_m6'], row['hours_m5'], row['hours_m4'],
                row['hours_m3'], row['hours_m2'], row['hours_m1']]
        
        if sum(hours) == 0:
            return 0.0
        
        # Linear regression: y = ax + b
        try:
            slope, _ = np.polyfit(months, hours, 1)
            return slope
        except:
            return 0.0
    
    df_eng['trend_slope_abs'] = df_eng.apply(calculate_trend_slope, axis=1)
    
    # Relative trend slope
    df_eng['trend_slope'] = df_eng['trend_slope_abs'] / (df_eng['AVG_L6M_HOURS'] + 1e-6)
    df_eng['trend_slope'] = df_eng['trend_slope'].fillna(0)
    
    # ========== Predicted Viewing Drop ==========
    # Predict next month hours
    df_eng['predicted_next'] = df_eng['hours_m1'] * (1 + df_eng['trend_slope'])
    df_eng['predicted_next'] = df_eng['predicted_next'].clip(lower=0)
    
    # Calculate percentage drop
    df_eng['PREDICTED_VIEWING_DROP_PCT'] = (
        (1 - df_eng['predicted_next'] / (df_eng['AVG_L6M_HOURS'] + 1e-6)) * 100
    )
    df_eng['PREDICTED_VIEWING_DROP_PCT'] = df_eng['PREDICTED_VIEWING_DROP_PCT'].fillna(0)
    
    # ========== Categorical Encoding ==========
    # Device Type
    df_eng['DEVICE_MOBILE'] = (df_eng['device_type'] == 'mobile').astype(int)
    df_eng['DEVICE_TV'] = (df_eng['device_type'] == 'tv').astype(int)
    df_eng['DEVICE_WEB'] = (df_eng['device_type'] == 'web').astype(int)
    
    # Plan Type
    df_eng['PLAN_BASIC'] = (df_eng['plan_type'] == 'basic').astype(int)
    df_eng['PLAN_STANDARD'] = (df_eng['plan_type'] == 'standard').astype(int)
    df_eng['PLAN_PREMIUM'] = (df_eng['plan_type'] == 'premium').astype(int)
    
    # Region
    df_eng['REGION_NORTH'] = (df_eng['region'] == 'north').astype(int)
    df_eng['REGION_CENTRAL'] = (df_eng['region'] == 'central').astype(int)
    df_eng['REGION_SOUTH'] = (df_eng['region'] == 'south').astype(int)
    
    return df_eng


def validate_input(data):
    """
    Validate input data
    
    Args:
        data: dict or DataFrame with customer data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = [
        'hours_m1', 'hours_m2', 'hours_m3', 
        'hours_m4', 'hours_m5', 'hours_m6',
        'tenure_months', 'is_promo_subscriber',
        'device_type', 'plan_type', 'region'
    ]
    
    # Check required fields
    if isinstance(data, dict):
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"Missing fields: {', '.join(missing)}"
        
        # Validate hours >= 0
        for i in range(1, 7):
            if data[f'hours_m{i}'] < 0:
                return False, f"hours_m{i} must be >= 0"
        
        # Validate tenure
        if data['tenure_months'] < 1:
            return False, "tenure_months must be >= 1"
        
        # Validate categorical
        valid_devices = ['mobile', 'tv', 'web']
        if data['device_type'] not in valid_devices:
            return False, f"device_type must be one of {valid_devices}"
        
        valid_plans = ['basic', 'standard', 'premium']
        if data['plan_type'] not in valid_plans:
            return False, f"plan_type must be one of {valid_plans}"
        
        valid_regions = ['north', 'central', 'south']
        if data['region'] not in valid_regions:
            return False, f"region must be one of {valid_regions}"
    
    return True, ""


def get_feature_names():
    """Return list of all engineered feature names"""
    return [
        'hours_m6', 'trend_slope_abs', 'is_promo_subscriber', 'tenure_months',
        'STDDEV_L3M_HOURS', 'GROWTH_RATE_L1M_VS_L3M', 'GROWTH_RATE_L3M_VS_L6M',
        'PREDICTED_VIEWING_DROP_PCT', 'CV_L3M_HOURS', 'DEVICE_MOBILE',
        'DEVICE_TV', 'DEVICE_WEB', 'PLAN_BASIC', 'PLAN_STANDARD',
        'PLAN_PREMIUM', 'REGION_NORTH', 'REGION_CENTRAL', 'REGION_SOUTH', 'trend_slope'
    ]
