"""
Model Loader Module
Safely loads and validates ML models
"""

import joblib
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load ML model from file
    
    Args:
        model_path: Path to model file (.pkl)
        
    Returns:
        dict: Model information including model object and features
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            # New format with metadata
            # Handle both 'model' and 'model_object' keys
            model_obj = model_data.get('model') or model_data.get('model_object')
            
            model_info = {
                'model_object': model_obj,
                'model_name': model_data.get('model_name', 'Unknown'),
                'features': model_data.get('features', get_default_features()),
                'scaler': model_data.get('scaler', None),
                'metadata': {
                    'train_accuracy': model_data.get('train_accuracy'),
                    'test_accuracy': model_data.get('test_accuracy'),
                    'f1_score': model_data.get('f1_score'),
                    'roc_auc': model_data.get('roc_auc'),
                    'training_date': model_data.get('training_date'),
                    'best_params': model_data.get('best_params')
                }
            }
        else:
            # Old format - just the model object
            model_info = {
                'model_object': model_data,
                'model_name': type(model_data).__name__,
                'features': get_default_features(),
                'scaler': None,
                'metadata': {}
            }
        
        # Validate model
        validate_model(model_info)
        
        logger.info(f"✅ Model loaded successfully: {model_info['model_name']}")
        logger.info(f"   Features: {len(model_info['features'])}")
        if model_info['metadata'].get('test_accuracy'):
            logger.info(f"   Test Accuracy: {model_info['metadata']['test_accuracy']:.4f}")
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


def validate_model(model_info):
    """
    Validate loaded model
    
    Args:
        model_info: dict with model information
        
    Raises:
        ValueError: if model is invalid
    """
    # Check model object
    if model_info['model_object'] is None:
        raise ValueError("Model object is None")
    
    # Check required methods
    required_methods = ['predict', 'predict_proba']
    for method in required_methods:
        if not hasattr(model_info['model_object'], method):
            raise ValueError(f"Model missing required method: {method}")
    
    # Check features
    if not model_info['features']:
        raise ValueError("No features specified")
    
    # Fix sklearn compatibility issue (monotonic_cst attribute)
    model = model_info['model_object']
    if hasattr(model, 'estimators_'):
        for estimator in model.estimators_:
            if not hasattr(estimator, 'monotonic_cst'):
                estimator.monotonic_cst = None
    
    logger.info("✅ Model validation passed")


def get_default_features():
    """
    Return default feature list for FPTPlay churn model
    
    Returns:
        list: Feature names
    """
    return [
        'hours_m6',
        'trend_slope_abs',
        'is_promo_subscriber',
        'tenure_months',
        'STDDEV_L3M_HOURS',
        'GROWTH_RATE_L1M_VS_L3M',
        'GROWTH_RATE_L3M_VS_L6M',
        'PREDICTED_VIEWING_DROP_PCT',
        'CV_L3M_HOURS',
        'DEVICE_MOBILE',
        'DEVICE_TV',
        'DEVICE_WEB',
        'PLAN_BASIC',
        'PLAN_STANDARD',
        'PLAN_PREMIUM',
        'REGION_NORTH',
        'REGION_CENTRAL',
        'REGION_SOUTH',
        'trend_slope'
    ]


def save_model(model, features, model_name, output_path, scaler=None, metadata=None):
    """
    Save model with metadata
    
    Args:
        model: trained model object
        features: list of feature names
        model_name: name of the model
        output_path: path to save model
        scaler: optional scaler object
        metadata: optional metadata dict
    """
    model_data = {
        'model_object': model,
        'model_name': model_name,
        'features': features,
        'scaler': scaler,
        'metadata': metadata or {}
    }
    
    joblib.dump(model_data, output_path)
    logger.info(f"✅ Model saved to: {output_path}")
