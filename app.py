"""
FPTPlay Churn Prediction Web Application
Flask Backend - Enterprise Edition

Features:
- Single customer prediction
- Batch CSV upload prediction
- Dashboard with statistics
- Export results to CSV/PDF
- RESTful API endpoints
- Logging and error handling
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
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', config['app']['secret_key'])
app.config['MAX_CONTENT_LENGTH'] = config['upload']['max_file_size_mb'] * 1024 * 1024
app.config['UPLOAD_FOLDER'] = config['upload']['upload_folder']

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# ========== LOAD MODEL ==========
try:
    model_path = config['model']['model_path']
    model_info = load_model(model_path)
    logger.info(f"✅ Model loaded successfully: {model_info['model_name']}")
    logger.info(f"   Features: {len(model_info['features'])} features")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    logger.error(traceback.format_exc())
    model_info = None

# ========== GLOBAL VARIABLES ==========
# Store prediction results in memory (for demo purposes)
prediction_history = []

# ========== UTILITY FUNCTIONS ==========

def calculate_risk_level(probability):
    """Calculate risk level from churn probability"""
    if probability > 0.7:
        return 'HIGH', '#c62828', 'Rủi ro cao - Cần hành động ngay'
    elif probability > 0.4:
        return 'MEDIUM', '#f57c00', 'Rủi ro trung bình - Theo dõi sát'
    else:
        return 'LOW', '#2e7d32', 'Rủi ro thấp - An toàn'

def predict_single(customer_data):
    """
    Predict churn for a single customer
    
    Args:
        customer_data: dict with customer features
        
    Returns:
        dict with prediction results
    """
    try:
        # Check if model is loaded
        if model_info is None:
            raise Exception("Model chưa được load. Vui lòng kiểm tra logs.")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Feature engineering
        df_eng = feature_engineering(df)
        
        # Select features
        X = df_eng[model_info['features']]
        
        # Predict
        model = model_info['model_object']
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        # Get feature importance
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for feat, imp in zip(model_info['features'], importances):
                feature_importance.append({
                    'feature': feat,
                    'importance': float(imp),
                    'value': float(X[feat].values[0])
                })
            # Sort by importance
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
        
        # Calculate risk level
        risk_level, risk_color, risk_message = calculate_risk_level(probability)
        
        result = {
            'churn_prediction': 'YES' if prediction == 1 else 'NO',
            'churn_probability': float(probability),
            'churn_probability_pct': f"{probability * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_message': risk_message,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_single: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_batch(df):
    """
    Predict churn for multiple customers
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        DataFrame with predictions
    """
    try:
        # Check if model is loaded
        if model_info is None:
            raise Exception("Model chưa được load. Vui lòng kiểm tra logs.")
        
        # Feature engineering
        df_eng = feature_engineering(df)
        
        # Select features
        X = df_eng[model_info['features']]
        
        # Predict
        model = model_info['model_object']
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
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
        
        return df_result
        
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ========== WEB ROUTES ==========

@app.route('/')
def index():
    """Homepage with prediction form"""
    return render_template('index.html', config=config, model_loaded=model_info is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single customer prediction"""
    try:
        # Check if model is loaded
        if model_info is None:
            return render_template('error.html', 
                                 error='Model chưa được load. Vui lòng kiểm tra cấu hình server.'), 500
        
        # Get form data
        customer_data = {
            'hours_m1': float(request.form.get('hours_m1', 0)),
            'hours_m2': float(request.form.get('hours_m2', 0)),
            'hours_m3': float(request.form.get('hours_m3', 0)),
            'hours_m4': float(request.form.get('hours_m4', 0)),
            'hours_m5': float(request.form.get('hours_m5', 0)),
            'hours_m6': float(request.form.get('hours_m6', 0)),
            'tenure_months': int(request.form.get('tenure_months', 12)),
            'is_promo_subscriber': int(request.form.get('is_promo_subscriber', 0)),
            'device_type': request.form.get('device_type', 'mobile'),
            'plan_type': request.form.get('plan_type', 'basic'),
            'region': request.form.get('region', 'south')
        }
        
        # Validate inputs
        if any(customer_data[f'hours_m{i}'] < 0 for i in range(1, 7)):
            return jsonify({'error': 'Giờ xem không được âm'}), 400
        
        if customer_data['tenure_months'] < 1:
            return jsonify({'error': 'Thời gian sử dụng phải >= 1 tháng'}), 400
        
        # Predict
        result = predict_single(customer_data)
        
        # Save to history
        history_entry = {
            'customer_data': customer_data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.append(history_entry)
        
        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        logger.info(f"Prediction completed: {result['churn_prediction']} ({result['churn_probability_pct']})")
        
        return render_template('result.html', 
                             customer_data=customer_data,
                             result=result,
                             config=config)
        
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error=str(e)), 500

@app.route('/batch-upload', methods=['GET', 'POST'])
def batch_upload():
    """Batch prediction from CSV file"""
    if request.method == 'GET':
        return render_template('batch_upload.html', config=config, model_loaded=model_info is not None)
    
    try:
        # Check if model is loaded
        if model_info is None:
            return jsonify({'error': 'Model chưa được load'}), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Chỉ chấp nhận file CSV'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
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
            'churn_count': (df_result['churn_prediction'] == 'YES').sum(),
            'no_churn_count': (df_result['churn_prediction'] == 'NO').sum(),
            'churn_rate': f"{(df_result['churn_prediction'] == 'YES').sum() / len(df_result) * 100:.1f}%",
            'avg_probability': f"{df_result['churn_probability'].mean() * 100:.1f}%",
            'high_risk': (df_result['risk_level'] == 'HIGH').sum(),
            'medium_risk': (df_result['risk_level'] == 'MEDIUM').sum(),
            'low_risk': (df_result['risk_level'] == 'LOW').sum()
        }
        
        logger.info(f"Batch prediction completed: {len(df_result)} customers")
        
        return render_template('batch_result.html',
                             stats=stats,
                             results=df_result.to_dict('records')[:50],  # Show first 50
                             config=config)
        
    except Exception as e:
        logger.error(f"Error in /batch-upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    try:
        # Calculate statistics from history
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
            results = [entry['result'] for entry in prediction_history]
            churn_count = sum(1 for r in results if r['churn_prediction'] == 'YES')
            total = len(results)
            
            stats = {
                'total_predictions': total,
                'churn_count': churn_count,
                'no_churn_count': total - churn_count,
                'churn_rate': f"{churn_count / total * 100:.1f}%" if total > 0 else '0.0%',
                'avg_probability': f"{sum(r['churn_probability'] for r in results) / total * 100:.1f}%" if total > 0 else '0.0%',
                'high_risk': sum(1 for r in results if r['risk_level'] == 'HIGH'),
                'medium_risk': sum(1 for r in results if r['risk_level'] == 'MEDIUM'),
                'low_risk': sum(1 for r in results if r['risk_level'] == 'LOW')
            }
        
        return render_template('dashboard.html', 
                             stats=stats,
                             history=prediction_history[-10:],  # Last 10 predictions
                             config=config,
                             model_loaded=model_info is not None)
        
    except Exception as e:
        logger.error(f"Error in /dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error=str(e)), 500

@app.route('/export/<format>')
def export_results(format):
    """Export results to CSV or JSON"""
    try:
        if not prediction_history:
            return jsonify({'error': 'Chưa có dữ liệu để export'}), 400
        
        # Prepare data
        data = []
        for entry in prediction_history:
            row = entry['customer_data'].copy()
            row.update(entry['result'])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8-sig')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        
        elif format == 'json':
            output = io.StringIO()
            df.to_json(output, orient='records', indent=2)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=f'churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        else:
            return jsonify({'error': 'Format không hợp lệ'}), 400
        
    except Exception as e:
        logger.error(f"Error in /export: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ========== API ENDPOINTS ==========

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """RESTful API endpoint for prediction"""
    try:
        # Check if model is loaded
        if model_info is None:
            return jsonify({
                'success': False,
                'error': 'Model chưa được load. Vui lòng kiểm tra cấu hình server.'
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Predict
        result = predict_single(data)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error in /api/predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info is not None,
        'timestamp': datetime.now().isoformat()
    })

# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error='Trang không tồn tại'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error='Lỗi server nội bộ'), 500

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File quá lớn'}), 413

# ========== MAIN ==========

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting FPTPlay Churn Prediction App")
    logger.info(f"   Port: {port}")
    logger.info(f"   Debug: {debug}")
    logger.info(f"   Model: {'Loaded ✅' if model_info else 'Not Loaded ❌'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
