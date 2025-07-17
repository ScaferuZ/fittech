#!/usr/bin/env python3
"""
XGFitness AI Visualization Runner
Run this script after training to generate comprehensive visualizations
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🎨 XGFitness AI Visualization Generator")
    print("="*60)
    
    # Check for trained models
    model_files = [
        'models/research_model_comparison.pkl',
        'models/xgfitness_ai_model.pkl'
    ]
    
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("   Please run 'python train_model.py' first")
        return
    
    print(f"📁 Found {len(available_models)} trained model(s):")
    for i, model_file in enumerate(available_models, 1):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"   {i}. {model_file} ({size_mb:.1f} MB)")
    
    # Prioritize research model for visualizations (has both XGBoost + Random Forest)
    if 'models/research_model_comparison.pkl' in available_models:
        model_to_use = 'models/research_model_comparison.pkl'
        print(f"\n🚀 Using RESEARCH model (best for visualizations): {model_to_use}")
    else:
        # Fallback to production model
        model_to_use = available_models[0]
        print(f"\n🚀 Using model: {model_to_use}")
    
    # Load the model
    try:
        print("📥 Loading trained model...")
        
        # Import the actual model class
        from src.thesis_model import XGFitnessAIModel
        
        # Create a proper model instance
        model = XGFitnessAIModel('../data')
        
        # Load the model data
        with open(model_to_use, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all attributes from the saved model
        for key, value in model_data.items():
            setattr(model, key, value)
        
        # Ensure is_trained is set correctly
        model.is_trained = getattr(model, 'is_trained', False) or (
            hasattr(model, 'workout_model') and model.workout_model is not None
        )
        
        # Ensure all expected attributes exist (even if None)
        for attr in [
            'workout_model', 'nutrition_model', 'workout_rf_model', 'nutrition_rf_model',
            'scaler', 'rf_scaler', 'workout_label_encoder', 'nutrition_label_encoder',
            'workout_rf_label_encoder', 'nutrition_rf_label_encoder',
            'training_info', 'rf_training_info', 'workout_templates', 'nutrition_templates'
        ]:
            if not hasattr(model, attr):
                setattr(model, attr, None)
        
        print("✅ Model loaded successfully")
        
        # Print model info
        print(f"   - Model type: {getattr(model, 'model_type', 'Unknown')}")
        print(f"   - Model version: {getattr(model, 'model_version', 'Unknown')}")
        print(f"   - Training samples: {model.training_info.get('training_samples', 'Unknown')}")
        print(f"   - Has XGBoost models: {'✅' if hasattr(model, 'workout_model') else '❌'}")
        print(f"   - Has Random Forest models: {'✅' if hasattr(model, 'workout_rf_model') and model.workout_rf_model else '❌'}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate or load training data for visualizations
    try:
        # Try to load existing training data
        training_data_files = [
            'training_data.csv',
            'data/training_data.csv',
            '../data/training_data.csv'
        ]
        
        df_training = None
        for data_file in training_data_files:
            if os.path.exists(data_file):
                print(f"📊 Loading training data from: {data_file}")
                df_training = pd.read_csv(data_file)
                break
        
        if df_training is None:
            print("⚠️  No training data CSV found, generating sample data...")
            df_training = generate_sample_data_for_viz(model)
            
        print(f"✅ Training data ready: {len(df_training)} samples")
        
    except Exception as e:
        print(f"⚠️  Error loading training data: {e}")
        print("   Generating sample data for visualization...")
        df_training = generate_sample_data_for_viz(model)
    
    # Generate visualizations
    try:
        print(f"\n🎨 Generating comprehensive visualizations...")
        
        # Import and run the visualization suite
        from visualisations import XGFitnessIndividualVisualizationSuite
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"visualizations/run_{timestamp}"
        
        # Generate all visualizations
        viz_suite = XGFitnessIndividualVisualizationSuite(model, df_training, output_dir)
        
        # Check if we're using actual training data with all features
        using_real_data = df_training is not None and len(df_training.columns) >= 15
        if using_real_data:
            print("✅ Using real training data - generating complete visualizations")
        else:
            print("⚠️  Using limited data - some visualizations may be simplified")
            
        viz_suite.generate_all_individual_charts()

        # === Additional custom visualizations for thesis ===
        try:
            print("\n🆕 Generating additional thesis visualizations...")
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. BMI Category vs Workout Template heatmap
            if 'bmi_category' in df_training.columns and 'workout_template_id' in df_training.columns:
                pivot = pd.pivot_table(
                    df_training,
                    index='bmi_category',
                    columns='workout_template_id',
                    aggfunc='size',
                    fill_value=0
                )
                plt.figure(figsize=(10,6))
                sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
                plt.title('Workout Template Assignment by BMI Category')
                plt.ylabel('BMI Category')
                plt.xlabel('Workout Template ID')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/bmi_category_vs_workout_template.png')
                plt.close()

            # 2. Goal vs Activity vs Template heatmap
            if all(col in df_training.columns for col in ['fitness_goal', 'activity_level', 'workout_template_id']):
                pivot2 = pd.pivot_table(
                    df_training,
                    index=['fitness_goal', 'activity_level'],
                    columns='workout_template_id',
                    aggfunc='size',
                    fill_value=0
                )
                plt.figure(figsize=(14,7))
                sns.heatmap(pivot2, annot=True, fmt='d', cmap='coolwarm')
                plt.title('Workout Template by Goal and Activity Level')
                plt.ylabel('Goal, Activity Level')
                plt.xlabel('Workout Template ID')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/goal_activity_vs_workout_template.png')
                plt.close()

            # 3. Feature importance bar charts (XGBoost and RF)
            def plot_feature_importance(model_obj, feature_names, title, filename):
                if hasattr(model_obj, 'feature_importances_'):
                    importances = model_obj.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    plt.figure(figsize=(12,6))
                    plt.title(title)
                    plt.bar(range(len(importances)), importances[indices], align='center')
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/{filename}')
                    plt.close()

            # Get feature names from the model
            feature_names = [
                'age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'tdee', 'activity_multiplier', 
                'Mod_act', 'Vig_act', 'age_bmi_interaction', 'tdee_per_kg', 'activity_intensity', 
                'height_weight_ratio', 'bmr_per_kg', 'age_activity_interaction', 'gender_Male', 
                'age_group_young', 'age_group_middle', 'age_group_older', 'bmi_Normal', 
                'bmi_Overweight', 'bmi_Obese', 'bmi_low_normal', 'bmi_high_normal', 
                'bmi_low_overweight', 'bmi_high_overweight', 'activity_High Activity', 
                'activity_Moderate Activity', 'activity_Low Activity'
            ]

            # Plot XGBoost feature importance
            if hasattr(model, 'workout_model') and model.workout_model is not None:
                plot_feature_importance(
                    model.workout_model, feature_names, 
                    'XGBoost Workout Model - Feature Importance',
                    'xgb_workout_feature_importance.png'
                )
                
            if hasattr(model, 'nutrition_model') and model.nutrition_model is not None:
                plot_feature_importance(
                    model.nutrition_model, feature_names, 
                    'XGBoost Nutrition Model - Feature Importance',
                    'xgb_nutrition_feature_importance.png'
                )

            # Plot Random Forest feature importance
            if hasattr(model, 'workout_rf_model') and model.workout_rf_model is not None:
                plot_feature_importance(
                    model.workout_rf_model, feature_names, 
                    'Random Forest Workout Model - Feature Importance',
                    'rf_workout_feature_importance.png'
                )
                
            if hasattr(model, 'nutrition_rf_model') and model.nutrition_rf_model is not None:
                plot_feature_importance(
                    model.nutrition_rf_model, feature_names, 
                    'Random Forest Nutrition Model - Feature Importance',
                    'rf_nutrition_feature_importance.png'
                )

            print("✅ Additional thesis visualizations generated")

        except Exception as e:
            print(f"⚠️  Error generating additional visualizations: {e}")

        print(f"\n✅ All visualizations completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated {len(os.listdir(output_dir))} visualization files")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

def generate_sample_data_for_viz(model):
    """Generate sample data for visualization when real data is not available"""
    print("Generating sample data for demonstration...")
    
    np.random.seed(42)
    n_samples = 1000
    
    df_training = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'height_cm': np.random.normal(170, 10, n_samples),
        'weight_kg': np.random.normal(70, 15, n_samples),
        'activity_level': np.random.choice(['Low Activity', 'Moderate Activity', 'High Activity'], 
                                         n_samples, p=[0.3, 0.4, 0.3]),
        'fitness_goal': np.random.choice(['Fat Loss', 'Muscle Gain', 'Maintenance'], 
                                       n_samples, p=[0.5, 0.3, 0.2]),
        'data_source': np.random.choice(['real', 'synthetic'], n_samples, p=[0.7, 0.3]),
        'split': np.random.choice(['train', 'validation', 'test'], n_samples, p=[0.7, 0.15, 0.15]),
        'workout_template_id': np.random.randint(1, 10, n_samples),
        'nutrition_template_id': np.random.randint(1, 9, n_samples),
        'Mod_act': np.random.uniform(0, 10, n_samples),
        'Vig_act': np.random.uniform(0, 5, n_samples)
    })
    
    # Calculate derived fields
    df_training['bmi'] = df_training['weight_kg'] / ((df_training['height_cm'] / 100) ** 2)
    
    # Add BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df_training['bmi_category'] = df_training['bmi'].apply(categorize_bmi)
    
    return df_training

if __name__ == "__main__":
    main()
