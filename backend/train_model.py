#!/usr/bin/env python3
"""
XGFitness AI Model Training Script
Trains DUAL XGBoost models (main AI) + DUAL Random Forest models (comparison)
Implements EXACT user requirements for thesis authenticity
"""

import os
import sys
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from thesis_model import XGFitnessAIModel

def main():
    """Train XGBoost + Random Forest models with STRICT AUTHENTICITY"""
    print("=" * 80)
    print("🏋️ FITTECH AI MODEL TRAINING - DUAL MODEL SYSTEM")
    print("=" * 80)
    print("STREAMLINED TRAINING PIPELINE:")
    print("1. 🚀 PRODUCTION Model: XGBoost-only (web application)")
    print("2. 📊 RESEARCH Model: XGBoost + Random Forest (thesis analysis)")  
    print("3. ✅ Reproducible results with fixed random seeds")
    print("4. 📈 Comprehensive visualizations for thesis")
    print("5. 🎯 Exact data splitting: 70% train / 15% val / 15% test")
    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize model
    print("📚 Initializing XGFitness AI Model...")
    model = XGFitnessAIModel(templates_dir='../data')
    print("✅ Model initialized successfully")
    print()
    
    # Create training dataset with EXACT AUTHENTICITY METHODOLOGY
    print("🔍 Loading real data with EXACT AUTHENTICITY METHODOLOGY...")
    training_data, test_df_original = model.load_real_data_with_exact_splits()
    print(f"✅ Training dataset loaded: {len(training_data)} samples")
    
    # Analyze data limitations transparently (no artificial fixes)
    print("\n📋 Running transparent data limitations analysis...")
    model.analyze_data_limitations(training_data)
    print(f"✅ Data limitations analysis completed")
    
    # Save training data for visualizations
    print("💾 Saving training data for visualizations...")
    training_data_path = 'training_data.csv'
    training_data.to_csv(training_data_path, index=False)
    print(f"✅ Training data saved to: {training_data_path}")
    print()
    
    # Train ALL models with verification
    print("🚀 Starting COMPREHENSIVE model training...")
    start_time = time.time()
    
    # Train all models using the unified training method
    print("\n🚀 Training ALL Models (XGBoost + Random Forest)...")
    comprehensive_info = model.train_all_models(training_data, test_df_original, random_state=42)
    
    training_time = time.time() - start_time
    print(f"✅ COMPREHENSIVE training completed in {training_time:.2f} seconds")
    print()
    
    # Extract training info
    xgboost_info = comprehensive_info['xgb_training_info']
    random_forest_info = comprehensive_info['rf_training_info']
    
    # Display comprehensive training results
    print("📊 COMPREHENSIVE TRAINING RESULTS:")
    print("=" * 80)
    print(f"Dataset Information:")
    print(f"  Total samples: {xgboost_info['total_samples']}")
    print(f"  Training samples: {xgboost_info['training_samples']}")
    print(f"  Validation samples: {xgboost_info['validation_samples']}")
    print(f"  Test samples: {xgboost_info['test_samples']}")
    print()
    
    print(f"XGBOOST MODEL PERFORMANCE (Main AI for Web App):")
    print(f"  Workout Model:")
    print(f"    - Accuracy: {xgboost_info['workout_accuracy']:.4f}")
    print(f"    - F1 Score: {xgboost_info['workout_f1']:.4f}")
    print(f"    - Precision: {xgboost_info.get('workout_precision', 'N/A'):.4f}")
    print(f"    - Recall: {xgboost_info.get('workout_recall', 'N/A'):.4f}")
    print(f"  Nutrition Model:")
    print(f"    - Accuracy: {xgboost_info['nutrition_accuracy']:.4f}")
    print(f"    - F1 Score: {xgboost_info['nutrition_f1']:.4f}")
    print(f"    - Precision: {xgboost_info.get('nutrition_precision', 'N/A'):.4f}")
    print(f"    - Recall: {xgboost_info.get('nutrition_recall', 'N/A'):.4f}")
    print()
    
    print(f"RANDOM FOREST MODEL PERFORMANCE (Academic Comparison):")
    print(f"  Workout Model:")
    print(f"    - Accuracy: {random_forest_info['rf_workout_accuracy']:.4f}")
    print(f"    - F1 Score: {random_forest_info['rf_workout_f1']:.4f}")
    print(f"    - Precision: {random_forest_info.get('rf_workout_precision', 'N/A'):.4f}")
    print(f"    - Recall: {random_forest_info.get('rf_workout_recall', 'N/A'):.4f}")
    print(f"  Nutrition Model:")
    print(f"    - Accuracy: {random_forest_info['rf_nutrition_accuracy']:.4f}")
    print(f"    - F1 Score: {random_forest_info['rf_nutrition_f1']:.4f}")
    print(f"    - Precision: {random_forest_info.get('rf_nutrition_precision', 'N/A'):.4f}")
    print(f"    - Recall: {random_forest_info.get('rf_nutrition_recall', 'N/A'):.4f}")
    print()
    
    # Model comparison summary
    print("🔍 MODEL COMPARISON SUMMARY:")
    print("-" * 40)
    xgb_workout_acc = xgboost_info['workout_accuracy']
    rf_workout_acc = random_forest_info['rf_workout_accuracy']
    xgb_nutrition_acc = xgboost_info['nutrition_accuracy']
    rf_nutrition_acc = random_forest_info['rf_nutrition_accuracy']
    
    workout_winner = "XGBoost" if xgb_workout_acc > rf_workout_acc else "Random Forest"
    nutrition_winner = "XGBoost" if xgb_nutrition_acc > rf_nutrition_acc else "Random Forest"
    
    print(f"Workout Model Winner: {workout_winner} ({max(xgb_workout_acc, rf_workout_acc):.4f})")
    print(f"Nutrition Model Winner: {nutrition_winner} ({max(xgb_nutrition_acc, rf_nutrition_acc):.4f})")
    print()
    
    # Save models using the streamlined TWO-MODEL approach
    print("💾 Saving models using DUAL-MODEL strategy...")
    print()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 1. PRODUCTION MODEL: XGBoost-only (for web application)
    print("🚀 Saving PRODUCTION model (XGBoost-only for web app)...")
    production_path = 'models/xgfitness_ai_model.pkl'
    model.save_model(production_path, include_research_models=False)  # XGBoost only
    
    # Get file size
    prod_size = os.path.getsize(production_path) / (1024 * 1024)  # Size in MB
    print(f"✅ PRODUCTION model saved: {production_path}")
    print(f"   - File size: {prod_size:.2f} MB")
    print(f"   - Contains: XGBoost models only (optimized for web app)")
    print(f"   - XGBoost Workout Accuracy: {xgboost_info['workout_accuracy']:.1%}")
    print(f"   - XGBoost Nutrition Accuracy: {xgboost_info['nutrition_accuracy']:.1%}")
    print()
    
    # 2. RESEARCH MODEL: Both algorithms (for thesis analysis)
    print("📊 Saving RESEARCH model (XGBoost + Random Forest for thesis)...")
    research_path = 'models/research_model_comparison.pkl'
    model.save_model(research_path, include_research_models=True)  # Both algorithms
    
    # Get file size
    research_size = os.path.getsize(research_path) / (1024 * 1024)  # Size in MB
    print(f"✅ RESEARCH model saved: {research_path}")
    print(f"   - File size: {research_size:.2f} MB") 
    print(f"   - Contains: XGBoost + Random Forest models (complete analysis)")
    print(f"   - Random Forest Workout Accuracy: {random_forest_info['rf_workout_accuracy']:.1%}")
    print(f"   - Random Forest Nutrition Accuracy: {random_forest_info['rf_nutrition_accuracy']:.1%}")
    print()
    
    print("🎯 DUAL-MODEL SUMMARY:")
    print(f"   📱 Production model: {prod_size:.1f}MB (web-ready)")
    print(f"   📊 Research model: {research_size:.1f}MB (thesis-ready)")
    print(f"   🔄 Reproducible results with random_state=42")
    print()
    
    # Generate comprehensive visualizations
    print("🎨 Generating comprehensive visualizations...")
    print("⚠️ Visualization generation skipped during training")
    print("   Run visualizations separately: python run_visualizations.py")
    print("   (This ensures training data is available for visualizations)")
    print()
    
    # Extract test set for honest evaluation (real data only)
    print("\n🔍 Running final model comparison on real test set (rule-based = upper bound)...")
    X_test = comprehensive_info['X_test']
    y_workout_test = comprehensive_info['y_workout_test']
    y_nutrition_test = comprehensive_info['y_nutrition_test']
    test_indices = comprehensive_info['test_indices']
    test_df = comprehensive_info['test_df_original']
    X_test_scaled = comprehensive_info['X_test_scaled']
    
    # Get the processed test data that matches the ML model evaluation
    X, y_workout, y_nutrition, df_enhanced = model.prepare_training_data(test_df)
    test_mask = df_enhanced['split'] == 'test'
    if test_mask.sum() == 0:
        # If no test split found, use all data
        processed_test_df = df_enhanced
    else:
        processed_test_df = df_enhanced[test_mask]
    
    # This will print and save the summary table
    model.report_model_comparison(X_test, y_workout_test, y_nutrition_test, processed_test_df, test_indices, output_path='model_comparison_summary.csv')
    print("\n[INFO] Rule-based system is the theoretical upper bound. ML models are expected to underperform due to data imbalance and limited real data. This summary is saved for thesis reporting.\n")
    
    # Print comprehensive thesis comparison summary
    print("\n📊 Generating comprehensive thesis comparison summary...")
    from thesis_model import print_model_comparison_summary
    
    # Create models dictionary for the summary function
    workout_models = {
        'xgboost': model.workout_model,
        'random_forest': model.workout_rf_model
    }
    nutrition_models = {
        'xgboost': model.nutrition_model,
        'random_forest': model.nutrition_rf_model
    }
    
    # Call the comprehensive summary function
    print_model_comparison_summary(
        workout_models, nutrition_models, test_indices, test_df, 
        y_workout_test, y_nutrition_test, X_test_scaled
    )
    
    print()
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds")
    print()
    print("🚀 Your DUAL model system is now ready:")
    print("   ✅ DUAL XGBoost models (main AI for web app)")
    print("   ✅ DUAL Random Forest models (academic comparison)")
    print("   ✅ Authentic data splitting methodology")
    print("   ✅ Preserved population characteristics")
    print("   ✅ Enhanced confidence scoring")
    print()
    print("📊 NEXT STEPS:")
    print("   1. Generate comprehensive visualizations:")
    print("      python run_visualizations.py")
    print("   2. Run the complete pipeline:")
    print("      .\\train_and_visualize.bat")
    print("   3. Restart your Flask app to use the new models:")
    print("      python app.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
