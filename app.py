"""
FPTPlay Churn Prediction Web Application
Flask Backend - Enterprise Edition with Full Error Handling

Features:
- Single customer prediction
- Batch CSV upload prediction  
- Dashboard with statistics
- Export results to CSV/PDF
- RESTful API endpoints
- Comprehensive logging and error handling
- Safe None-checking everywhere
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
import os
import io
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import yaml
import traceback

# Import utilities
from utils.feature_engineering import feature_engineering
from utils.model_loader import load_model, validate_model

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)

# Load configuration
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    config = {
        'app': {'secret_key': 'dev-secret-key'},
        'upload': {'max_file_size_mb': 10, 'upload_folder': 'uploads'}
    }

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', config.get('app', {}).get('secret_key', 'fallback-secret'))
app.config['MAX_CONTENT_LENGTH'] = config.get('upload', {}).get('max_file_size_mb', 10) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = config.get('upload', {}).get('upload_folder', 'uploads')

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== LOGGING SETUP ==========
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== LOAD MODEL ==========
model_info = None

try:
    model_path = config.get('model', {}).get('model_path', './best_model_random_forest.pkl')
    logger.info(f"🔄 Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        logger.error(f"   Current directory: {os.getcwd()}")
        logger.error(f"   Files in directory: {os.listdir('.')}")
    else:
        model_info = load_model(model_path)
        
        if model_info is None:
            logger.error("❌ load_model returned None")
        elif not isinstance(model_info, dict):
            logger.error(f"❌ load_model returned unexpected type: {type(model_info)}")
        elif model_info.get('model_object') is None:
            logger.error("❌ model_object is None in model_info")
        elif not model_info.get('features'):
            logger.error("❌ features list is empty in model_info")
        else:
            logger.info(f"✅ Model loaded successfully: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"   Model type: {type(model_info.get('model_object'))}")
            logger.info(f"   Features: {len(model_info.get('features', []))} features")
            logger.info(f"   Feature list: {model_info.get('features', [])}")
            
except Exception as e:
    logger.error(f"❌ CRITICAL: Failed to load model: {str(e)}")
    logger.error(f"   Exception type: {type(e)}")
    logger.error(f"   Traceback: {traceback.format_exc()}")
    model_info = None

# ========== GLOBAL VARIABLES ==========
prediction_history = []

# ========== UTILITY FUNCTIONS ==========

def calculate_risk_level(probability):
    """Calculate risk level from churn probability - SAFE"""
    try:
        prob = float(probability) if probability is not None else 0.0
        
        if prob > 0.7:
            return 'HIGH', '#c62828', 'Rủi ro cao - Cần hành động ngay'
        elif prob > 0.4:
            return 'MEDIUM', '#f57c00', 'Rủi ro trung bình - Theo dõi sát'
        else:
            return 'LOW', '#2e7d32', 'Rủi ro thấp - An toàn'
    except Exception as e:
        logger.error(f"Error in calculate_risk_level: {e}")
        return 'UNKNOWN', '#9e9e9e', 'Không xác định'


def predict_single(customer_data):
    """
    Predict churn for a single customer - WITH FULL VALIDATION
    """
    try:
        # CRITICAL CHECK 1: Model loaded?
        if model_info is None:
            error_msg = "Model chưa được load. Vui lòng kiểm tra logs server."
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        # CRITICAL CHECK 2: customer_data valid?
        if customer_data is None:
            error_msg = "customer_data is None"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        if not isinstance(customer_data, dict):
            error_msg = f"customer_data must be dict, got {type(customer_data)}"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        # Log input data
        logger.info(f"📊 Prediction input: {json.dumps(customer_data, indent=2)}")
        
        # CRITICAL CHECK 3: model_object exists?
        model_obj = model_info.get('model_object')
        if model_obj is None:
            error_msg = "model_object is None in model_info"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        # CRITICAL CHECK 4: features list exists?
        features = model_info.get('features')
        if not features:
            error_msg = "features list is empty or None"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        logger.info(f"✅ Created DataFrame with shape: {df.shape}")
        
        # Feature engineering
        df_eng = feature_engineering(df)
        logger.info(f"✅ Feature engineering completed. Shape: {df_eng.shape}")
        logger.info(f"   Engineered columns: {list(df_eng.columns)}")
        
        # CRITICAL CHECK 5: All required features present?
        missing_features = [f for f in features if f not in df_eng.columns]
        if missing_features:
            error_msg = f"Missing features after engineering: {missing_features}"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        # Select features
        X = df_eng[features]
        logger.info(f"✅ Selected features. Shape: {X.shape}")
        
        # Predict
        prediction = model_obj.predict(X)
        if prediction is None or len(prediction) == 0:
            error_msg = "model.predict() returned None or empty"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        prediction_value = int(prediction[0])
        logger.info(f"✅ Prediction: {prediction_value}")
        
        # Predict probability
        proba = model_obj.predict_proba(X)
        if proba is None or len(proba) == 0:
            error_msg = "model.predict_proba() returned None or empty"
            logger.error(f"❌ predict_single failed: {error_msg}")
            raise ValueError(error_msg)
        
        probability = float(proba[0, 1])
        logger.info(f"✅ Probability: {probability:.4f}")
        
        # Get feature importance
        feature_importance = []
        try:
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                for feat, imp in zip(features, importances):
                    feature_importance.append({
                        'feature': feat,
                        'importance': float(imp),
                        'value': float(X[feat].values[0])
                    })
                feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            feature_importance = []
        
        # Calculate risk level
        risk_level, risk_color, risk_message = calculate_risk_level(probability)
        
        result = {
            'churn_prediction': 'YES' if prediction_value == 1 else 'NO',
            'churn_probability': probability,
            'churn_probability_pct': f"{probability * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_message': risk_message,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction completed successfully: {result['churn_prediction']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ ERROR in predict_single: {str(e)}")
        logger.error(f"   Exception type: {type(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise


def predict_batch(df):
    """Predict churn for multiple customers - WITH VALIDATION"""
    try:
        # Check model
        if model_info is None:
            raise ValueError("Model chưa được load")
        
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        logger.info(f"📊 Batch prediction for {len(df)} customers")
        
        # Feature engineering
        df_eng = feature_engineering(df)
        
        # Select features
        features = model_info.get('features')
        if not features:
            raise ValueError("Features list is empty")
        
        X = df_eng[features]
        
        # Predict
        model_obj = model_info.get('model_object')
        if model_obj is None:
            raise ValueError("model_object is None")
        
        predictions = model_obj.predict(X)
        probabilities = model_obj.predict_proba(X)[:, 1]
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['churn_prediction'] = ['YES' if p == 1 else 'NO' for p in predictions]
        df_result['churn_probability'] = probabilities
        df_result['churn_probability_pct'] = [f"{p * 100:.1f}%" for p in probabilities]
        
        # Add risk levels
        risk_data = [calculate_risk_level(p) for p in probabilities]
        df_result['risk_level'] = [r[0] for r in risk_data]
        df_result['risk_color'] = [r[1] for r in risk_data]
        df_result['risk_message'] = [r[2] for r in risk_data]
        
        logger.info(f"✅ Batch prediction completed")
        return df_result
        
    except Exception as e:
        logger.error(f"❌ ERROR in predict_batch: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ========== WEB ROUTES ==========

@app.route('/')
def index():
    """Homepage with prediction form"""
    try:
        return render_template('index.html', 
                             config=config, 
                             model_loaded=(model_info is not None))
    except Exception as e:
        logger.error(f"Error in /: {e}")
        return f"Error loading homepage: {str(e)}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """Handle single customer prediction"""
    try:
        # Check model first
        if model_info is None:
            return render_template('error.html', 
                                 error='⚠️ Model chưa được load. Vui lòng liên hệ admin.'), 503
        
        # Get form data - WITH DEFAULTS
        customer_data = {
            'hours_m1': float(request.form.get('hours_m1', 0) or 0),
            'hours_m2': float(request.form.get('hours_m2', 0) or 0),
            'hours_m3': float(request.form.get('hours_m3', 0) or 0),
            'hours_m4': float(request.form.get('hours_m4', 0) or 0),
            'hours_m5': float(request.form.get('hours_m5', 0) or 0),
            'hours_m6': float(request.form.get('hours_m6', 0) or 0),
            'tenure_months': int(request.form.get('tenure_months', 12) or 12),
            'is_promo_subscriber': int(request.form.get('is_promo_subscriber', 0) or 0),
            'device_type': request.form.get('device_type', 'mobile') or 'mobile',
            'plan_type': request.form.get('plan_type', 'basic') or 'basic',
            'region': request.form.get('region', 'south') or 'south'
        }
        
        logger.info(f"📝 Form data received: {customer_data}")
        
        # Validate inputs
        if any(customer_data[f'hours_m{i}'] < 0 for i in range(1, 7)):
            return render_template('error.html', error='Giờ xem không được âm'), 400
        
        if customer_data['tenure_months'] < 1:
            return render_template('error.html', error='Thời gian sử dụng phải >= 1 tháng'), 400
        
        # Predict
        result = predict_single(customer_data)
        
        # Save to history
        history_entry = {
            'customer_data': customer_data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.append(history_entry)
        
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        logger.info(f"✅ Prediction completed: {result['churn_prediction']}")
        
        return render_template('result.html', 
                             customer_data=customer_data,
                             result=result,
                             config=config)
        
    except Exception as e:
        logger.error(f"❌ ERROR in /predict: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error=f"Lỗi dự đoán: {str(e)}"), 500


@app.route('/batch-upload', methods=['GET', 'POST'])
def batch_upload():
    """Batch prediction from CSV file"""
    if request.method == 'GET':
        return render_template('batch_upload.html', 
                             config=config, 
                             model_loaded=(model_info is not None))
    
    try:
        if model_info is None:
            return jsonify({'error': 'Model chưa được load'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
        
        file = request.files['file']
        
        if not file or file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Chỉ chấp nhận file CSV'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'File CSV trống'}), 400
        
        # Validate required columns
        required_cols = ['hours_m1', 'hours_m2', 'hours_m3', 'hours_m4', 'hours_m5', 'hours_m6']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Thiếu cột: {", ".join(missing_cols)}'}), 400
        
        # Predict
        df_result = predict_batch(df)
        
        # Save results to session
        session['batch_result'] = df_result.to_json(orient='records')
        session['batch_timestamp'] = datetime.now().isoformat()
        
        # Calculate statistics
        stats = {
            'total': len(df_result),
            'churn_count': int((df_result['churn_prediction'] == 'YES').sum()),
            'no_churn_count': int((df_result['churn_prediction'] == 'NO').sum()),
            'churn_rate': f"{(df_result['churn_prediction'] == 'YES').sum() / len(df_result) * 100:.1f}%",
            'avg_probability': f"{df_result['churn_probability'].mean() * 100:.1f}%",
            'high_risk': int((df_result['risk_level'] == 'HIGH').sum()),
            'medium_risk': int((df_result['risk_level'] == 'MEDIUM').sum()),
            'low_risk': int((df_result['risk_level'] == 'LOW').sum())
        }
        
        logger.info(f"✅ Batch prediction completed: {len(df_result)} customers")
        
        return render_template('batch_result.html',
                             stats=stats,
                             results=df_result.to_dict('records')[:50],
                             config=config)
        
    except Exception as e:
        logger.error(f"❌ ERROR in /batch-upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    try:
        if not prediction_history:
            stats = {
                'total_predictions': 0,
                'churn_count': 0,
                'no_churn_count': 0,
                'churn_rate': '0.0%',
                'avg_probability': '0.0%',
                'high_risk': 0,
                'medium_risk': 0,
                'low_risk': 0
            }
        else:
            results = [entry.get('result', {}) for entry in prediction_history if entry.get('result')]
            churn_count = sum(1 for r in results if r.get('churn_prediction') == 'YES')
            total = len(results)
            
            stats = {
                'total_predictions': total,
                'churn_count': churn_count,
                'no_churn_count': total - churn_count,
                'churn_rate': f"{churn_count / total * 100:.1f}%" if total > 0 else '0.0%',
                'avg_probability': f"{sum(r.get('churn_probability', 0) for r in results) / total * 100:.1f}%" if total > 0 else '0.0%',
                'high_risk': sum(1 for r in results if r.get('risk_level') == 'HIGH'),
                'medium_risk': sum(1 for r in results if r.get('risk_level') == 'MEDIUM'),
                'low_risk': sum(1 for r in results if r.get('risk_level') == 'LOW')
            }
        
        return render_template('dashboard.html', 
                             stats=stats,
                             history=prediction_history[-10:],
                             config=config,
                             model_loaded=(model_info is not None))
        
    except Exception as e:
        logger.error(f"❌ ERROR in /dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error=str(e)), 500


# ========== API ENDPOINTS ==========

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """RESTful API endpoint for prediction - WITH FULL VALIDATION"""
    try:
        # Check model
        if model_info is None:
            logger.error("API /api/predict called but model is None")
            return jsonify({
                'success': False,
                'error': 'Model chưa được load. Vui lòng kiểm tra cấu hình server.'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            logger.error("API /api/predict called with no data")
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        logger.info(f"API /api/predict called with data: {json.dumps(data, indent=2)}")
        
        # Predict
        result = predict_single(data)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"❌ ERROR in /api/predict: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    is_model_loaded = model_info is not None
    
    health_status = {
        'status': 'healthy' if is_model_loaded else 'degraded',
        'model_loaded': is_model_loaded,
        'timestamp': datetime.now().isoformat()
    }
    
    if is_model_loaded:
        health_status['model_name'] = model_info.get('model_name', 'Unknown')
        health_status['features_count'] = len(model_info.get('features', []))
    
    logger.info(f"Health check: {health_status}")
    
    return jsonify(health_status)


# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error='Trang không tồn tại'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"500 error: {e}")
    return render_template('error.html', error='Lỗi server nội bộ'), 500


@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File quá lớn'}), 413


# ========== MAIN ==========

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info("=" * 70)
    logger.info("🚀 STARTING FPTPLAY CHURN PREDICTION APP")
    logger.info("=" * 70)
    logger.info(f"   Port: {port}")
    logger.info(f"   Debug: {debug}")
    logger.info(f"   Model Status: {'✅ Loaded' if model_info else '❌ Not Loaded'}")
    if model_info:
        logger.info(f"   Model Name: {model_info.get('model_name', 'Unknown')}")
        logger.info(f"   Features: {len(model_info.get('features', []))}")
    logger.info("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
