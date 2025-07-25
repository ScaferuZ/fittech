"""
Enhanced Flask API for XGFitness AI system
Provides comprehensive fitness recommendations with confidence scoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime
import logging
import numpy as np

# Add src directory to path for imports
backend_dir = os.path.dirname(__file__)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

# Also add to PYTHONPATH for pickle
import sys
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from thesis_model import XGFitnessAIModel
from validation import validate_api_request_data, create_validation_summary, get_validation_rules
from calculations import (calculate_bmr, calculate_tdee, categorize_bmi, 
                         calculate_complete_nutrition_plan, verify_daily_totals)
from templates import get_template_manager
from src.meal_plan_calculator import MealPlanCalculator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize meal plan calculator
meal_plan_calculator = MealPlanCalculator()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

def initialize_model():
    """Initialize or load the XGFitness AI model"""
    global model
    
    try:
        model = XGFitnessAIModel('../data')  # Use main data directory
        
        # Try to load existing model
        model_path = 'models/xgfitness_ai_model.pkl'
        if os.path.exists(model_path):
            model.load_model(model_path)
            logger.info("Loaded existing trained model")
        else:
            # Train new model if none exists
            logger.info("No existing model found, training new model...")
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            training_data = model.create_training_dataset(total_samples=2000)
            model.train_models(training_data)
            model.save_model(model_path)
            
            # Save templates using template manager
            model.template_manager.save_all_templates()
            
            logger.info("Model trained and saved successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'FitTech AI API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Get fitness recommendations',
            '/health': 'GET - Check API health',
            '/templates': 'GET - Get available templates',
            '/validation-rules': 'GET - Get input validation rules',
            '/calculate-metrics': 'POST - Calculate BMI, BMR, TDEE',
            '/improve-recommendation': 'POST - Get improved recommendations based on feedback',
            '/meal-plan': 'POST - Generate daily meal plan including snacks',
            '/meal-options/<meal_type>': 'GET - Get available meal options for a specific meal type',
            '/scale-meal': 'POST - Scale a specific meal to target calories',
            '/weekly-meal-plan': 'POST - Generate a 7-day meal plan with variety including snacks',
            '/calculate-complete-nutrition': 'POST - Complete 8-step nutrition calculation pipeline',
            '/nutrition-from-model': 'POST - Get complete nutrition plan using model prediction'
        },
        'documentation': 'Send POST request to /predict with user data'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model
    
    status = {
        'status': 'healthy' if model and model.is_trained else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'model_trained': model.is_trained if model else False
    }
    
    if model and model.is_trained:
        # Convert NumPy types to native Python types for JSON serialization
        training_info = convert_numpy_types(model.training_info)
        status.update(training_info)
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for fitness recommendations
    
    Expected JSON input:
    {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity",
        "fitness_goal": "Muscle Gain"
    }
    """
    global model
    
    try:
        # Check if model is available
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available. Please try again later.',
                'code': 'MODEL_UNAVAILABLE'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'code': 'NO_DATA'
            }), 400
        
        # Validate input data
        try:
            clean_data = validate_api_request_data(data)
            validation_warnings = []
        except Exception as validation_error:
            return jsonify({
                'success': False,
                'error': 'Invalid input data',
                'validation_error': str(validation_error),
                'code': 'VALIDATION_ERROR'
            }), 400
        
        # Make prediction
        prediction_result = model.predict_with_confidence(clean_data)
        
        # Add validation warnings if any
        if validation_warnings:
            prediction_result['validation_warnings'] = validation_warnings
        
        # Add request metadata
        prediction_result['request_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model.training_info.get('training_date', 'Unknown'),
            'api_version': '1.0.0'
        }
        
        # Convert NumPy types to native Python types for JSON serialization
        prediction_result = convert_numpy_types(prediction_result)
        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction',
            'code': 'PREDICTION_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/calculate-metrics', methods=['POST'])
def calculate_metrics():
    """
    Calculate basic fitness metrics (BMI, BMR, TDEE) without full recommendations
    
    Expected JSON input:
    {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity"
    }
    """
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Basic validation for required fields
        required_fields = ['age', 'gender', 'height', 'weight', 'activity_level']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Calculate metrics
        age = int(data['age'])
        gender = data['gender']
        height = float(data['height'])
        weight = float(data['weight'])
        activity_level = data['activity_level']
        
        # Validate ranges
        if not (18 <= age <= 100):
            return jsonify({'success': False, 'error': 'Age must be between 18 and 100'}), 400
        if not (120 <= height <= 250):
            return jsonify({'success': False, 'error': 'Height must be between 120 and 250 cm'}), 400
        if not (30 <= weight <= 300):
            return jsonify({'success': False, 'error': 'Weight must be between 30 and 300 kg'}), 400
        
        bmi = weight / ((height / 100) ** 2)
        bmi_category = categorize_bmi(bmi)
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        
        return jsonify({
            'success': True,
            'metrics': {
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error calculating metrics'
        }), 500

@app.route('/templates', methods=['GET'])
def get_templates():
    """Get all available workout and nutrition templates"""
    global model
    
    try:
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        return jsonify({
            'success': True,
            'workout_templates': model.template_manager.workout_templates.to_dict('records'),
            'nutrition_templates': model.template_manager.nutrition_templates.to_dict('records'),
            'template_count': {
                'workout': len(model.template_manager.workout_templates),
                'nutrition': len(model.template_manager.nutrition_templates)
            }
        })
        
    except Exception as e:
        logger.error(f"Templates error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving templates'
        }), 500

@app.route('/validation-rules', methods=['GET'])
def get_validation_rules_endpoint():
    """Get validation rules for frontend form validation"""
    try:
        rules = get_validation_rules()
        return jsonify({
            'success': True,
            'validation_rules': rules
        })
    except Exception as e:
        logger.error(f"Validation rules error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving validation rules'
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information and training statistics"""
    global model
    
    try:
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        return jsonify({
            'success': True,
            'model_info': model.training_info,
            'feature_count': len(model.feature_columns),
            'template_counts': {
                'workout_templates': len(model.template_manager.workout_templates),
                'nutrition_templates': len(model.template_manager.nutrition_templates)
            }
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving model information'
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the model with new parameters (admin endpoint)
    """
    global model
    
    try:
        # This could be protected with authentication in production
        data = request.get_json() or {}
        
        n_samples = data.get('n_samples', 2000)
        
        if not (500 <= n_samples <= 10000):
            return jsonify({
                'success': False,
                'error': 'n_samples must be between 500 and 10000'
            }), 400
        
        logger.info(f"Starting model retraining with {n_samples} samples...")
        
        # Reinitialize model
        model = XGFitnessAIModel('../data')  # Use main data directory
        
        # Generate new training data
        training_data = model.create_training_dataset(
            real_data_file='../e267_Data on age, gender, height, weight, activity levels for each household member.txt',
            total_samples=n_samples
        )
        
        # Train model
        training_results = model.train_models(training_data)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save_model('models/xgfitness_ai_model.pkl')
        
        logger.info("Model retraining completed successfully")
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'training_results': {
                'workout_accuracy': training_results['workout_accuracy'],
                'nutrition_accuracy': training_results['nutrition_accuracy'],
                'training_samples': n_samples
            },
            'model_info': model.training_info
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Error during model retraining'
        }), 500

@app.route('/improve-recommendation', methods=['POST'])
def improve_recommendation():
    """
    Generate improved recommendations based on user feedback
    
    Expected JSON input:
    {
        "currentRecommendation": {...},
        "userProfile": {...},
        "feedback": {
            "workoutDifficulty": "too_hard",
            "workoutEnjoyment": "enjoyed",
            "workoutEffectiveness": "effective",
            "nutritionSatisfaction": "satisfied",
            "energyLevel": "good",
            "recovery": "good",
            "overallSatisfaction": "satisfied"
        }
    }
    """
    global model
    
    try:
        # Check if model is available
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available. Please try again later.',
                'code': 'MODEL_UNAVAILABLE'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'code': 'NO_DATA'
            }), 400
        
        current_recommendation = data.get('currentRecommendation', {})
        user_profile = data.get('userProfile', {})
        feedback = data.get('feedback', {})
        
        if not current_recommendation or not user_profile or not feedback:
            return jsonify({
                'success': False,
                'error': 'Missing required data: currentRecommendation, userProfile, or feedback',
                'code': 'MISSING_DATA'
            }), 400
        
        # Analyze feedback and generate suggestions
        suggestions = analyze_feedback_and_suggest_improvements(
            current_recommendation, user_profile, feedback, model
        )
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error improving recommendation: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Internal server error during recommendation improvement',
            'code': 'IMPROVEMENT_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/meal-plan', methods=['POST'])
def generate_meal_plan():
    """Generate daily meal plan including snacks based on user requirements"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract required parameters
        target_calories = data.get('target_calories')
        target_protein = data.get('target_protein')
        target_carbs = data.get('target_carbs') 
        target_fat = data.get('target_fat')
        preferences = data.get('preferences', {})
        
        if not target_calories:
            return jsonify({'error': 'target_calories is required'}), 400
        
        # Set defaults for macros if not provided
        if not target_protein:
            target_protein = int(target_calories * 0.25 / 4)  # 25% of calories from protein
        if not target_carbs:
            target_carbs = int(target_calories * 0.45 / 4)   # 45% of calories from carbs
        if not target_fat:
            target_fat = int(target_calories * 0.30 / 9)     # 30% of calories from fat
        
        # Generate meal plan
        meal_plan = meal_plan_calculator.calculate_daily_meal_plan(
            target_calories=int(target_calories),
            target_protein=int(target_protein),
            target_carbs=int(target_carbs),
            target_fat=int(target_fat),
            preferences=preferences
        )
        
        if meal_plan.get('success'):
            return jsonify({
                'success': True,
                'meal_plan': meal_plan,
                'message': 'Meal plan generated successfully including snacks'
            })
        else:
            return jsonify({
                'success': False,
                'error': meal_plan.get('error', 'Failed to generate meal plan')
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating meal plan: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/meal-options/<meal_type>', methods=['GET'])
def get_meal_options(meal_type):
    """Get available meal options for a specific meal type (sarapan, makan_siang, makan_malam, snack)"""
    try:
        # Validate meal type
        valid_types = ['sarapan', 'makan_siang', 'makan_malam', 'snack', 'camilan']
        if meal_type not in valid_types:
            return jsonify({
                'error': f'Invalid meal type. Must be one of: {valid_types}'
            }), 400
        
        # Convert camilan to snack for backend compatibility
        backend_meal_type = 'snack' if meal_type == 'camilan' else meal_type
        
        options = meal_plan_calculator.get_meal_options(backend_meal_type)
        
        return jsonify({
            'success': True,
            'meal_type': meal_type,
            'options': options
        })
        
    except Exception as e:
        logger.error(f"Error getting meal options: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/scale-meal', methods=['POST'])
def scale_meal():
    """Scale a specific meal to target calories"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        meal_id = data.get('meal_id')
        target_calories = data.get('target_calories')
        
        if not meal_id or not target_calories:
            return jsonify({
                'error': 'meal_id and target_calories are required'
            }), 400
        
        scaled_meal = meal_plan_calculator.calculate_single_meal(
            meal_id=meal_id,
            target_calories=float(target_calories)
        )
        
        if scaled_meal.get('success'):
            return jsonify({
                'success': True,
                'scaled_meal': scaled_meal
            })
        else:
            return jsonify({
                'success': False,
                'error': scaled_meal.get('error', 'Failed to scale meal')
            }), 500
            
    except Exception as e:
        logger.error(f"Error scaling meal: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/weekly-meal-plan', methods=['POST'])
def generate_weekly_meal_plan():
    """Generate a 7-day meal plan with variety including snacks"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        daily_calories = data.get('daily_calories')
        daily_protein = data.get('daily_protein')
        daily_carbs = data.get('daily_carbs')
        daily_fat = data.get('daily_fat')
        
        if not daily_calories:
            return jsonify({'error': 'daily_calories is required'}), 400
        
        # Set defaults for macros if not provided
        if not daily_protein:
            daily_protein = int(daily_calories * 0.25 / 4)
        if not daily_carbs:
            daily_carbs = int(daily_calories * 0.45 / 4)
        if not daily_fat:
            daily_fat = int(daily_calories * 0.30 / 9)
        
        weekly_plan = meal_plan_calculator.generate_weekly_meal_plan(
            daily_calories=int(daily_calories),
            daily_protein=int(daily_protein),
            daily_carbs=int(daily_carbs),
            daily_fat=int(daily_fat)
        )
        
        if weekly_plan.get('success'):
            return jsonify({
                'success': True,
                'weekly_plan': weekly_plan,
                'message': 'Weekly meal plan generated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': weekly_plan.get('error', 'Failed to generate weekly meal plan')
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating weekly meal plan: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/calculate-complete-nutrition', methods=['POST'])
def calculate_complete_nutrition():
    """
    Complete 8-step nutrition calculation pipeline:
    1. Calculate BMR
    2. Apply activity factor (TDEE)  
    3. Apply template caloric multiplier to get target calories
    4. Determine calculation weight
    5. Calculate each macronutrient using template multipliers
    6. Verify totals (macro calories should match target calories ±50)
    7. Generate meal plan according to output calories and macros
    8. Verify totals for a full day
    
    Expected JSON input:
    {
        "user_profile": {
            "age": 25,
            "gender": "Male", 
            "height": 175,
            "weight": 70,
            "activity_level": "Moderate Activity",
            "fitness_goal": "Muscle Gain"
        },
        "nutrition_template": {
            "template_id": 5,
            "goal": "Muscle Gain",
            "bmi_category": "Normal",
            "caloric_intake_multiplier": 1.10,
            "protein_per_kg": 2.1,
            "carbs_per_kg": 4.25,
            "fat_per_kg": 0.95
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_profile = data.get('user_profile')
        nutrition_template = data.get('nutrition_template')
        
        if not user_profile or not nutrition_template:
            return jsonify({
                'error': 'Both user_profile and nutrition_template are required'
            }), 400
        
        # Steps 1-6: Complete nutrition calculations
        nutrition_result = calculate_complete_nutrition_plan(user_profile, nutrition_template)
        
        if not nutrition_result['success']:
            return jsonify({
                'success': False,
                'error': nutrition_result['error']
            }), 400
        
        calculations = nutrition_result['calculations']
        
        # Step 7: Generate meal plan according to output calories and macros
        target_calories = calculations['target_calories']
        target_protein = calculations['macronutrients']['protein_g']
        target_carbs = calculations['macronutrients']['carbs_g']
        target_fat = calculations['macronutrients']['fat_g']
        
        meal_plan_result = meal_plan_calculator.calculate_daily_meal_plan(
            target_calories=target_calories,
            target_protein=int(target_protein),
            target_carbs=int(target_carbs),
            target_fat=int(target_fat),
            preferences=data.get('preferences', {})
        )
        
        # Step 8: Verify totals for a full day
        daily_verification = verify_daily_totals(
            meal_plan_result, target_calories, target_protein, target_carbs, target_fat
        )
        
        # Compile complete response
        complete_response = {
            'success': True,
            'nutrition_calculations': nutrition_result,
            'meal_plan': meal_plan_result,
            'daily_verification': daily_verification,
            'summary': {
                'step_1_bmr': calculations['bmr'],
                'step_2_tdee': calculations['tdee'],
                'step_3_target_calories': calculations['target_calories'],
                'step_4_calculation_weight': calculations['calculation_weight'],
                'step_5_macronutrients': calculations['macronutrients'],
                'step_6_verification': nutrition_result['verification'],
                'step_7_meal_plan_generated': meal_plan_result.get('success', False),
                'step_8_daily_verification': daily_verification.get('verified', False)
            }
        }
        
        return jsonify(complete_response)
        
    except Exception as e:
        logger.error(f"Error in complete nutrition calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/nutrition-from-model', methods=['POST'])
def nutrition_from_model():
    """
    Get complete nutrition plan using model prediction and template lookup
    
    Expected JSON input:
    {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity", 
        "fitness_goal": "Muscle Gain"
    }
    """
    global model
    
    try:
        # Check if model is available
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available. Please try again later.',
                'code': 'MODEL_UNAVAILABLE'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get model prediction to find the right nutrition template
        prediction_result = model.predict_with_confidence(data)
        
        if not prediction_result:
            return jsonify({
                'success': False,
                'error': 'Failed to get model prediction'
            }), 500
        
        # Extract nutrition template from prediction
        nutrition_template = prediction_result['predictions']['nutrition_template']
        
        # Use the complete nutrition calculation pipeline
        complete_nutrition = calculate_complete_nutrition_plan(data, nutrition_template)
        
        if not complete_nutrition['success']:
            return jsonify({
                'success': False,
                'error': complete_nutrition['error']
            }), 400
        
        calculations = complete_nutrition['calculations']
        
        # Generate meal plan
        meal_plan_result = meal_plan_calculator.calculate_daily_meal_plan(
            target_calories=calculations['target_calories'],
            target_protein=int(calculations['macronutrients']['protein_g']),
            target_carbs=int(calculations['macronutrients']['carbs_g']),
            target_fat=int(calculations['macronutrients']['fat_g']),
            preferences=data.get('preferences', {})
        )
        
        # Verify daily totals
        daily_verification = verify_daily_totals(
            meal_plan_result, 
            calculations['target_calories'],
            calculations['macronutrients']['protein_g'],
            calculations['macronutrients']['carbs_g'],
            calculations['macronutrients']['fat_g']
        )
        
        return jsonify({
            'success': True,
            'model_prediction': prediction_result,
            'nutrition_calculations': complete_nutrition,
            'meal_plan': meal_plan_result,
            'daily_verification': daily_verification
        })
        
    except Exception as e:
        logger.error(f"Error in nutrition from model: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }), 500

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # NumPy scalar
        return obj.item()
    else:
        return obj

def analyze_feedback_and_suggest_improvements(current_recommendation, user_profile, feedback, model):
    """
    Analyze user feedback and suggest improvements to recommendations
    """
    try:
        workout_changes = []
        nutrition_changes = []
        new_recommendation = current_recommendation.copy()
        
        # Analyze workout difficulty feedback
        if feedback.get('workoutDifficulty') == 'too_hard':
            workout_changes.append("Kurangi intensitas latihan")
            # Suggest easier template
            current_goal = current_recommendation.get('workout_recommendation', {}).get('goal', 'Maintenance')
            current_activity = current_recommendation.get('workout_recommendation', {}).get('activity_level', 'Moderate Activity')
            
            # Try to find easier template
            if current_activity == 'High Activity':
                new_activity = 'Moderate Activity'
            elif current_activity == 'Moderate Activity':
                new_activity = 'Low Activity'
            else:
                new_activity = current_activity
                
            new_template_id = model.template_manager.find_workout_template(current_goal, new_activity)
            if new_template_id:
                new_template = model.template_manager.get_workout_template(new_template_id)
                new_recommendation['workout_recommendation'] = new_template
                
        elif feedback.get('workoutDifficulty') == 'too_easy':
            workout_changes.append("Tingkatkan intensitas latihan")
            # Suggest harder template
            current_goal = current_recommendation.get('workout_recommendation', {}).get('goal', 'Maintenance')
            current_activity = current_recommendation.get('workout_recommendation', {}).get('activity_level', 'Low Activity')
            
            # Try to find harder template
            if current_activity == 'Low Activity':
                new_activity = 'Moderate Activity'
            elif current_activity == 'Moderate Activity':
                new_activity = 'High Activity'
            else:
                new_activity = current_activity
                
            new_template_id = model.template_manager.find_workout_template(current_goal, new_activity)
            if new_template_id:
                new_template = model.template_manager.get_workout_template(new_template_id)
                new_recommendation['workout_recommendation'] = new_template
        
        # Analyze workout enjoyment
        if feedback.get('workoutEnjoyment') == 'disliked':
            workout_changes.append("Pertimbangkan jenis latihan yang berbeda")
            # Suggest different workout type
            current_workout_type = current_recommendation.get('workout_recommendation', {}).get('workout_type', 'Full Body')
            if current_workout_type == 'Full Body':
                new_workout_type = 'Upper/Lower Split'
            elif current_workout_type == 'Upper/Lower Split':
                new_workout_type = 'Push/Pull/Legs'
            else:
                new_workout_type = 'Full Body'
            
            # Update workout type in recommendation
            if 'workout_recommendation' in new_recommendation:
                new_recommendation['workout_recommendation']['workout_type'] = new_workout_type
        
        # Analyze nutrition satisfaction
        if feedback.get('nutritionSatisfaction') == 'unsatisfied':
            nutrition_changes.append("Sesuaikan rencana nutrisi")
            # Suggest different nutrition approach
            current_goal = current_recommendation.get('nutrition_recommendation', {}).get('goal', 'Maintenance')
            current_bmi = user_profile.get('bmi_category', 'Normal')
            
            # Try different nutrition template
            available_templates = model.template_manager.nutrition_templates
            alternative_templates = available_templates[
                (available_templates['goal'] == current_goal) & 
                (available_templates['bmi_category'] != current_bmi)
            ]
            
            if not alternative_templates.empty:
                new_template = alternative_templates.iloc[0].to_dict()
                new_recommendation['nutrition_recommendation'] = new_template
        
        # Analyze energy and recovery
        if feedback.get('energyLevel') == 'low' or feedback.get('recovery') == 'poor':
            nutrition_changes.append("Tingkatkan nutrisi pemulihan")
            # Suggest higher protein or different macro ratios
            current_nutrition = new_recommendation.get('nutrition_recommendation', {})
            if current_nutrition:
                # Increase protein slightly
                current_nutrition['protein_per_kg'] = min(current_nutrition.get('protein_per_kg', 2.0) + 0.2, 3.0)
                # Recalculate targets
                weight = user_profile.get('weight', 70)
                current_nutrition['target_protein'] = int(weight * current_nutrition['protein_per_kg'])
        
        # Analyze overall satisfaction
        if feedback.get('overallSatisfaction') == 'unsatisfied':
            if not workout_changes:
                workout_changes.append("Pertimbangkan menyesuaikan frekuensi atau intensitas latihan")
            if not nutrition_changes:
                nutrition_changes.append("Tinjau target nutrisi")
        
        # Generate suggestion messages
        workout_changes_text = " dan ".join(workout_changes) if workout_changes else None
        nutrition_changes_text = " dan ".join(nutrition_changes) if nutrition_changes else None
        
        # Generate detailed reasoning
        reasoning_parts = []
        
        if feedback.get('workoutDifficulty') == 'too_hard':
            reasoning_parts.append("Latihan Anda terasa terlalu menantang, jadi saya telah menyarankan tingkat intensitas yang lebih mudah dikelola.")
        elif feedback.get('workoutDifficulty') == 'too_easy':
            reasoning_parts.append("Latihan Anda terasa terlalu mudah, jadi saya telah meningkatkan intensitas untuk membantu Anda berkembang lebih cepat.")
        
        if feedback.get('workoutEnjoyment') == 'disliked':
            reasoning_parts.append("Anda tidak menyukai jenis latihan saat ini, jadi saya telah menyarankan pendekatan yang berbeda.")
        
        if feedback.get('nutritionSatisfaction') == 'unsatisfied':
            reasoning_parts.append("Rencana nutrisi Anda tidak memuaskan, jadi saya telah menyesuaikan rasio makronutrien.")
        
        if feedback.get('energyLevel') == 'low' or feedback.get('recovery') == 'poor':
            reasoning_parts.append("Anda mengalami energi rendah atau pemulihan yang buruk, jadi saya telah mengoptimalkan nutrisi Anda untuk pemulihan yang lebih baik.")
        
        if feedback.get('overallSatisfaction') == 'unsatisfied':
            reasoning_parts.append("Secara keseluruhan, Anda tidak puas dengan rencana saat ini, jadi saya telah membuat penyesuaian yang komprehensif.")
        
        if not reasoning_parts:
            reasoning_parts.append("Berdasarkan feedback Anda, saya telah membuat penyesuaian yang ditargetkan untuk lebih sesuai dengan kebutuhan dan preferensi Anda.")
        
        reasoning = " ".join(reasoning_parts)
        
        return {
            'workoutChanges': workout_changes_text,
            'nutritionChanges': nutrition_changes_text,
            'newRecommendation': new_recommendation,
            'confidence': 'medium',
            'reasoning': reasoning
        }
        
    except Exception as e:
        logger.error(f"Error analyzing feedback: {str(e)}")
        return {
            'workoutChanges': "Tidak dapat menganalisis feedback latihan",
            'nutritionChanges': "Tidak dapat menganalisis feedback nutrisi", 
            'newRecommendation': current_recommendation,
            'confidence': 'low',
            'reasoning': 'Terjadi kesalahan selama analisis'
        }

def create_app():
    """Application factory"""
    # Initialize model
    if not initialize_model():
        logger.error("Failed to initialize model. API may not function properly.")
    
    return app

@app.route('/meal-plan-flexible', methods=['POST'])
def generate_flexible_meal_plan():
    """Generate daily meal plan with flexible individual food adjustments for best accuracy"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract required parameters
        target_calories = data.get('target_calories')
        target_protein = data.get('target_protein')
        target_carbs = data.get('target_carbs') 
        target_fat = data.get('target_fat')
        preferences = data.get('preferences', {})
        max_food_adjustment = data.get('max_food_adjustment', 0.4)  # Default ±40%
        
        if not target_calories:
            return jsonify({'error': 'target_calories is required'}), 400
        
        # Set defaults for macros if not provided
        if not target_protein:
            target_protein = int(target_calories * 0.25 / 4)  # 25% of calories from protein
        if not target_carbs:
            target_carbs = int(target_calories * 0.45 / 4)   # 45% of calories from carbs
        if not target_fat:
            target_fat = int(target_calories * 0.30 / 9)     # 30% of calories from fat
        
        # Validate max_food_adjustment
        if not (0.1 <= max_food_adjustment <= 1.0):
            return jsonify({'error': 'max_food_adjustment must be between 0.1 and 1.0'}), 400
        
        # Generate flexible meal plan
        meal_plan = meal_plan_calculator.calculate_daily_meal_plan(
            target_calories=int(target_calories),
            target_protein=int(target_protein),
            target_carbs=int(target_carbs),
            target_fat=int(target_fat),
            preferences=preferences,
            max_food_adjustment=max_food_adjustment
        )
        
        if meal_plan.get('success'):
            return jsonify({
                'success': True,
                'meal_plan': meal_plan,
                'message': f'Flexible meal plan generated with ±{int(max_food_adjustment*100)}% food adjustments'
            })
        else:
            return jsonify({
                'success': False,
                'error': meal_plan.get('error', 'Failed to generate flexible meal plan')
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating flexible meal plan: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Initialize model
    if initialize_model():
        print("FitTech AI Model initialized successfully")
        print("Starting Flask API server...")
        
        # Print available endpoints
        print("\nAvailable endpoints:")
        print("GET  /               - API documentation")
        print("GET  /health         - Health check")
        print("POST /predict        - Get fitness recommendations")
        print("POST /calculate-metrics - Calculate BMI, BMR, TDEE")
        print("GET  /templates      - Get available templates")
        print("GET  /validation-rules - Get validation rules")
        print("GET  /model-info     - Get model information")
        print("POST /retrain        - Retrain model (admin)")
        print("POST /improve-recommendation - Get improved recommendations based on feedback")
        print("POST /meal-plan      - Generate daily meal plan with snacks")
        print("GET  /meal-options/<meal_type> - Get meal options for specific type")
        print("POST /scale-meal     - Scale specific meal to target calories")
        print("POST /weekly-meal-plan - Generate 7-day meal plan")
        print("POST /meal-plan      - Generate daily meal plan including snacks")
        print("GET  /meal-options/<meal_type> - Get available meal options for a specific meal type")
        print("POST /scale-meal     - Scale a specific meal to target calories")
        print("POST /weekly-meal-plan - Generate a 7-day meal plan with variety including snacks")
        
        # Print example request
        print("\nExample request to /predict:")
        print("""{
    "age": 25,
    "gender": "Male",
    "height": 175,
    "weight": 70,
    "activity_level": "Moderate Activity",
    "fitness_goal": "Muscle Gain"
}""")
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize FitTech AI model. Exiting.")
        sys.exit(1)