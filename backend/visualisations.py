#!/usr/bin/env python3
"""
XGFitness AI Individual Chart Visualization Suite
Each chart is saved as a separate PNG file with side-by-side model comparisons
"""

import os
import sys
import warnings

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                               confusion_matrix, classification_report, roc_curve, auc)
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Visualization dependencies not available: {e}")
    print("   Install with: pip install matplotlib seaborn pandas numpy scikit-learn")
    VISUALIZATION_AVAILABLE = False
    # Create dummy imports to prevent errors
    plt = None
    sns = None
    pd = None
    np = None
    accuracy_score = lambda *args, **kwargs: 0.0
    f1_score = lambda *args, **kwargs: 0.0
    precision_score = lambda *args, **kwargs: 0.0
    recall_score = lambda *args, **kwargs: 0.0
    confusion_matrix = lambda *args, **kwargs: [[0]]
    classification_report = lambda *args, **kwargs: ""
    roc_curve = lambda *args, **kwargs: ([0], [0], [0])
    auc = lambda *args, **kwargs: 0.0
    label_binarize = lambda *args, **kwargs: [[0]]
    cycle = lambda *args: iter([])

from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set high-quality plotting parameters
if plt is not None:
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

# Rule-based baseline is now handled by the model's predict_rule_based() method
# This ensures consistency with the model's template assignment logic

class XGFitnessIndividualVisualizationSuite:
    """
    Individual chart visualization suite - each chart as separate PNG
    """
    
    def __init__(self, model, df_training, save_dir='visualizations_individual'):
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization dependencies not available. Install with: pip install matplotlib seaborn pandas numpy scikit-learn")
            
        self.model = model
        self.df_training = df_training
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Rule-based baseline is now handled by model.predict_rule_based()
        # No separate initialization needed
        
        # Professional color palette
        self.colors = {
            'xgboost': '#1f77b4',      # Professional blue
            'random_forest': '#ff7f0e', # Professional orange
            'rule_based': '#2ca02c',    # Professional green
            'real_data': '#2ca02c',    # Professional green
            'synthetic': '#d62728',    # Professional red
            'primary': '#3498db',      # Modern blue
            'secondary': '#e74c3c',    # Modern red
            'accent': '#f39c12',       # Modern orange
            'success': '#27ae60',      # Modern green
            'background': '#ecf0f1'    # Light background
        }
        
        # High-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        print(f"🎨 Individual Chart Suite initialized")
        print(f"   Save directory: {save_dir}")
        print(f"   Dataset size: {len(df_training)} samples")
        print(f"   Rule-based baseline: Using model.predict_rule_based()")
    
    def generate_all_individual_charts(self):
        """Generate all charts as individual PNG files"""
        print(f"\n🎨 GENERATING INDIVIDUAL CHARTS")
        print("="*80)
        
        # Dataset Composition Charts (6 individual charts)
        print("📊 1. Dataset composition charts...")
        self.create_data_source_pie()
        self.create_split_distribution()
        self.create_fitness_goal_distribution()
        self.create_activity_level_distribution()
        self.create_bmi_category_distribution()
        self.create_gender_distribution()
        
        # Data Quality Charts (4 individual charts)
        print("🔍 2. Data quality charts...")
        self.create_real_vs_synthetic_by_split()
        self.create_age_distribution()
        self.create_bmi_vs_tdee_scatter()
        self.create_activity_hours_scatter()
        
        # Demographic Charts (6 individual charts)
        print("👥 3. Demographic charts...")
        self.create_height_by_gender()
        self.create_weight_by_gender()
        self.create_bmi_distribution_with_categories()
        self.create_bmr_vs_age_by_gender()
        self.create_tdee_by_activity_level()
        self.create_age_vs_bmi_by_goal()
        
        # Template Analysis Charts (4 individual charts)
        print("📋 4. Template analysis charts...")
        self.create_workout_template_distribution()
        self.create_nutrition_template_distribution()
        self.create_fat_loss_activity_bmi_heatmap()
        self.create_top_template_combinations()
        
        # Model Performance Charts (if models trained)
        if self.model.is_trained:
            print("🤖 5. Model performance charts...")
            self.create_workout_model_comparison()
            self.create_nutrition_model_comparison()
            self.create_overall_model_performance()
            
            print("📈 6. Confusion matrices...")
            self.create_individual_confusion_matrices()
            
            print("📉 7. ROC curves...")
            self.create_individual_roc_curves()
            
            print("⚖️  8. Model comparison analysis...")
            self.create_prediction_agreement()
            self.create_algorithm_diversity()
        
        # Summary Charts
        print("📊 9. Summary charts...")
        self.create_dataset_summary()
        self.create_template_coverage()
        self.create_research_findings()
        
        print(f"\n✅ All individual charts completed!")
        self._list_individual_files()
    
    # Dataset Composition Individual Charts
    def create_data_source_pie(self):
        """Data source distribution pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        source_counts = self.df_training['data_source'].value_counts()
        colors_source = [self.colors['real_data'], self.colors['synthetic']]
        
        # Create explode array with correct length
        explode = [0.05] + [0] * (len(source_counts) - 1)
        
        wedges, texts, autotexts = ax.pie(
            source_counts.values, 
            labels=source_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=colors_source[:len(source_counts)],
            explode=explode
        )
        ax.set_title('Data Source Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/01_data_source_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_split_distribution(self):
        """Train/validation/test split distribution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        split_counts = self.df_training['split'].value_counts()
        bars = ax.bar(
            split_counts.index, 
            split_counts.values, 
            color=[self.colors['primary'], self.colors['accent'], self.colors['secondary']]
        )
        ax.set_title('Data Split Distribution', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/02_split_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_fitness_goal_distribution(self):
        """Fitness goal distribution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        goal_counts = self.df_training['fitness_goal'].value_counts()
        bars = ax.bar(
            goal_counts.index, 
            goal_counts.values, 
            color=[self.colors['success'], self.colors['accent'], self.colors['primary']]
        )
        ax.set_title('Fitness Goal Distribution', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels and percentages
        total = sum(goal_counts.values)
        for bar in bars:
            height = bar.get_height()
            pct = height / total * 100
            ax.annotate(f'{int(height)}\n({pct:.1f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/03_fitness_goal_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_activity_level_distribution(self):
        """Activity level distribution (Natural - Preserved)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        activity_counts = self.df_training['activity_level'].value_counts()
        bars = ax.bar(
            activity_counts.index, 
            activity_counts.values, 
            color=['#ff6b6b', '#4ecdc4', '#45b7d1']
        )
        ax.set_title('Activity Level Distribution\n(Natural - Preserved)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(activity_counts.values)
        for bar in bars:
            height = bar.get_height()
            pct = height / total * 100
            ax.annotate(f'{int(height)}\n({pct:.1f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/04_activity_level_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_bmi_category_distribution(self):
        """BMI category distribution (Natural - Preserved)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        bmi_counts = self.df_training['bmi_category'].value_counts()
        bars = ax.bar(
            bmi_counts.index, 
            bmi_counts.values, 
            color=['#ffeaa7', '#74b9ff', '#fd79a8', '#e17055']
        )
        ax.set_title('BMI Category Distribution\n(Natural - Preserved)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(bmi_counts.values)
        for bar in bars:
            height = bar.get_height()
            pct = height / total * 100
            ax.annotate(f'{int(height)}\n({pct:.1f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/05_bmi_category_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_gender_distribution(self):
        """Gender distribution pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        gender_counts = self.df_training['gender'].value_counts()
        colors_gender = [self.colors['primary'], self.colors['secondary']]
        ax.pie(
            gender_counts.values, 
            labels=gender_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=colors_gender
        )
        ax.set_title('Gender Distribution\n(Natural - Preserved)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/06_gender_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Data Quality Individual Charts
    def create_real_vs_synthetic_by_split(self):
        """Real vs synthetic data by split"""
        fig, ax = plt.subplots(figsize=(12, 8))
        split_source = self.df_training.groupby(['split', 'data_source']).size().unstack(fill_value=0)
        bars = split_source.plot(kind='bar', ax=ax, 
                                color=[self.colors['real_data'], self.colors['synthetic']], 
                                width=0.7)
        ax.set_title('Real vs Synthetic Data by Split\n(Validation & Test: 100% Real)', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Real Data', 'Synthetic Data'])
        
        # Add value labels on stacked bars
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/07_real_vs_synthetic_by_split.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_age_distribution(self):
        """Age distribution histogram"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(self.df_training['age'], bins=25, color=self.colors['primary'], 
               alpha=0.7, edgecolor='black')
        mean_age = self.df_training['age'].mean()
        ax.axvline(mean_age, color=self.colors['secondary'], 
                  linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f} years')
        ax.set_title('Age Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/08_age_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_bmi_vs_tdee_scatter(self):
        """BMI vs TDEE scatter plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            self.df_training['bmi'], 
            self.df_training['tdee'], 
            c=self.df_training['activity_level'].astype('category').cat.codes, 
            alpha=0.6, 
            cmap='viridis',
            s=30
        )
        ax.set_title('BMI vs TDEE by Activity Level', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI')
        ax.set_ylabel('TDEE (calories)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Activity Level')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/09_bmi_vs_tdee_scatter.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_activity_hours_scatter(self):
        """Moderate vs vigorous activity hours scatter"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(
            self.df_training['Mod_act'], 
            self.df_training['Vig_act'], 
            c=self.df_training['activity_level'].astype('category').cat.codes, 
            alpha=0.6, 
            cmap='viridis',
            s=30
        )
        ax.set_title('Moderate vs Vigorous Activity Hours', fontsize=16, fontweight='bold')
        ax.set_xlabel('Moderate Activity (hours/week)')
        ax.set_ylabel('Vigorous Activity (hours/week)')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/10_activity_hours_scatter.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Demographic Individual Charts
    def create_height_by_gender(self):
        """Height distribution by gender"""
        fig, ax = plt.subplots(figsize=(12, 8))
        for gender in ['Male', 'Female']:
            data = self.df_training[self.df_training['gender'] == gender]['height_cm']
            ax.hist(data, alpha=0.7, label=gender, bins=20)
        ax.set_title('Height Distribution by Gender', fontsize=16, fontweight='bold')
        ax.set_xlabel('Height (cm)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/11_height_by_gender.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_weight_by_gender(self):
        """Weight distribution by gender"""
        fig, ax = plt.subplots(figsize=(12, 8))
        for gender in ['Male', 'Female']:
            data = self.df_training[self.df_training['gender'] == gender]['weight_kg']
            ax.hist(data, alpha=0.7, label=gender, bins=20)
        ax.set_title('Weight Distribution by Gender', fontsize=16, fontweight='bold')
        ax.set_xlabel('Weight (kg)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/12_weight_by_gender.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_bmi_distribution_with_categories(self):
        """BMI distribution with WHO categories"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(self.df_training['bmi'], bins=25, color=self.colors['primary'], 
               alpha=0.7, edgecolor='black')
        
        # Add BMI category lines
        bmi_lines = [18.5, 25, 30]
        bmi_labels = ['Underweight|Normal', 'Normal|Overweight', 'Overweight|Obese']
        colors_lines = ['orange', 'red', 'darkred']
        
        for line, label, color in zip(bmi_lines, bmi_labels, colors_lines):
            ax.axvline(line, color=color, linestyle='--', alpha=0.8, label=label)
        
        ax.set_title('BMI Distribution with WHO Categories', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/13_bmi_distribution_with_categories.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_bmr_vs_age_by_gender(self):
        """BMR vs age colored by gender"""
        fig, ax = plt.subplots(figsize=(12, 8))
        for gender in ['Male', 'Female']:
            data = self.df_training[self.df_training['gender'] == gender]
            ax.scatter(data['age'], data['bmr'], alpha=0.6, 
                      label=gender, s=30)
        ax.set_title('BMR vs Age by Gender', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('BMR (calories)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/14_bmr_vs_age_by_gender.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_tdee_by_activity_level(self):
        """TDEE distribution by activity level"""
        fig, ax = plt.subplots(figsize=(12, 8))
        activity_levels = self.df_training['activity_level'].unique()
        for level in activity_levels:
            data = self.df_training[self.df_training['activity_level'] == level]['tdee']
            ax.hist(data, alpha=0.7, label=level, bins=15)
        ax.set_title('TDEE Distribution by Activity Level', fontsize=16, fontweight='bold')
        ax.set_xlabel('TDEE (calories)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/15_tdee_by_activity_level.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_age_vs_bmi_by_goal(self):
        """Age vs BMI colored by fitness goal"""
        fig, ax = plt.subplots(figsize=(12, 8))
        for goal in self.df_training['fitness_goal'].unique():
            data = self.df_training[self.df_training['fitness_goal'] == goal]
            ax.scatter(data['age'], data['bmi'], alpha=0.6, 
                      label=goal, s=30)
        ax.set_title('Age vs BMI by Fitness Goal', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('BMI')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/16_age_vs_bmi_by_goal.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Template Analysis Individual Charts
    def create_workout_template_distribution(self):
        """Workout template distribution"""
        fig, ax = plt.subplots(figsize=(12, 8))
        workout_counts = self.df_training['workout_template_id'].value_counts().sort_index()
        bars = ax.bar(
            workout_counts.index, 
            workout_counts.values, 
            color=self.colors['primary'],
            alpha=0.8
        )
        ax.set_title('Workout Template Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Workout Template ID')
        ax.set_ylabel('Number of Assignments')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/17_workout_template_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_nutrition_template_distribution(self):
        """Nutrition template distribution"""
        fig, ax = plt.subplots(figsize=(12, 8))
        nutrition_counts = self.df_training['nutrition_template_id'].value_counts().sort_index()
        bars = ax.bar(
            nutrition_counts.index, 
            nutrition_counts.values, 
            color=self.colors['success'],
            alpha=0.8
        )
        ax.set_title('Nutrition Template Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Nutrition Template ID')
        ax.set_ylabel('Number of Assignments')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/18_nutrition_template_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_fat_loss_activity_bmi_heatmap(self):
        """Fat loss: activity vs BMI heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        goal_data = self.df_training[self.df_training['fitness_goal'] == 'Fat Loss']
        if len(goal_data) > 0:
            pivot_table = goal_data.groupby(['activity_level', 'bmi_category']).size().unstack(fill_value=0)
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Fat Loss: Activity vs BMI\n({len(goal_data)} samples)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('BMI Category')
            ax.set_ylabel('Activity Level')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/19_fat_loss_activity_bmi_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_top_template_combinations(self):
        """Top 15 template combinations"""
        fig, ax = plt.subplots(figsize=(14, 8))
        template_combo = self.df_training.groupby(['workout_template_id', 'nutrition_template_id']).size()
        top_combos = template_combo.nlargest(15)
        
        bars = ax.bar(
            range(len(top_combos)), 
            top_combos.values, 
            color=self.colors['accent'],
            alpha=0.8
        )
        ax.set_title('Top 15 Template Combinations', fontsize=16, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Template Combination Rank')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            combo_idx = top_combos.index[i]
            ax.annotate(f'{int(height)}\n({combo_idx[0]},{combo_idx[1]})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/20_top_template_combinations.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Model Performance Side-by-Side Comparisons
    def create_workout_model_comparison(self):
        """Side-by-side workout model performance comparison with Rule-Based Baseline"""
        if not self.model.is_trained:
            return
            
        # Get test data
        test_data = self.df_training[self.df_training['split'] == 'test']
        if len(test_data) == 0:
            print("⚠️  No test data available for workout model comparison")
            return
        
        # Get actual labels
        y_w_test = test_data['workout_template_id'].values
        
        # Get Rule-Based Baseline predictions using model's method
        try:
            rule_w_pred, _ = self.model.predict_rule_based(test_data)
        except AttributeError:
            print("⚠️  Model does not have predict_rule_based method, skipping rule-based comparison")
            return
        
        # Prepare test data with proper feature engineering
        try:
            X, y_workout, y_nutrition, df_enhanced = self.model.prepare_training_data(self.df_training)
            test_mask = df_enhanced['split'] == 'test'
            X_test = X[test_mask]
            
            # Scale features for XGBoost
            X_test_xgb = self.model.scaler.transform(X_test)
            xgb_w_pred = self.model.workout_model.predict(X_test_xgb)
            
            # Convert back to template IDs
            xgb_w_pred_templates = self.model.workout_label_encoder.inverse_transform(xgb_w_pred)
            
            print(f"✅ XGBoost predictions obtained successfully")
        except Exception as e:
            print(f"⚠️  Error getting XGBoost predictions: {e}")
            return
        
        # Check RF availability and get predictions
        rf_available = (hasattr(self.model, 'workout_rf_model') and 
                       self.model.workout_rf_model is not None)
        
        if rf_available:
            try:
                X_test_rf = self.model.rf_scaler.transform(X_test)
                rf_w_pred = self.model.workout_rf_model.predict(X_test_rf)
                rf_w_pred_templates = self.model.workout_rf_label_encoder.inverse_transform(rf_w_pred)
                print(f"✅ RF predictions obtained successfully")
            except Exception as e:
                print(f"⚠️  Error getting RF predictions: {e}")
                rf_available = False
        
        # Calculate metrics for all models
        metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        # Rule-Based metrics
        rule_metrics = [
            accuracy_score(y_w_test, rule_w_pred),
            f1_score(y_w_test, rule_w_pred, average='weighted'),
            precision_score(y_w_test, rule_w_pred, average='weighted'),
            recall_score(y_w_test, rule_w_pred, average='weighted')
        ]
        
        # XGBoost metrics
        xgb_metrics = [
            accuracy_score(y_w_test, xgb_w_pred_templates),
            f1_score(y_w_test, xgb_w_pred_templates, average='weighted'),
            precision_score(y_w_test, xgb_w_pred_templates, average='weighted'),
            recall_score(y_w_test, xgb_w_pred_templates, average='weighted')
        ]
        
        # Create the comparison chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metrics_labels))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, rule_metrics, width, label='Rule-Based', color=self.colors['rule_based'], alpha=0.8)
        ax.bar(x, xgb_metrics, width, label='XGBoost', color=self.colors['xgboost'], alpha=0.8)
        
        if rf_available:
            # Random Forest metrics
            rf_metrics = [
                accuracy_score(y_w_test, rf_w_pred_templates),
                f1_score(y_w_test, rf_w_pred_templates, average='weighted'),
                precision_score(y_w_test, rf_w_pred_templates, average='weighted'),
                recall_score(y_w_test, rf_w_pred_templates, average='weighted')
            ]
            ax.bar(x + width, rf_metrics, width, label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Workout Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(rule_metrics):
            ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(xgb_metrics):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if rf_available:
            for i, v in enumerate(rf_metrics):
                ax.text(i + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/21_workout_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Workout model comparison chart created successfully")
    
    def create_nutrition_model_comparison(self):
        """Side-by-side nutrition model performance comparison with Rule-Based Baseline"""
        if not self.model.is_trained:
            return
            
        # Get test data
        test_data = self.df_training[self.df_training['split'] == 'test']
        if len(test_data) == 0:
            print("⚠️  No test data available for nutrition model comparison")
            return
        
        # Get actual labels
        y_n_test = test_data['nutrition_template_id'].values
        
        # Get Rule-Based Baseline predictions using model's method
        try:
            _, rule_n_pred = self.model.predict_rule_based(test_data)
        except AttributeError:
            print("⚠️  Model does not have predict_rule_based method, skipping rule-based comparison")
            return
        
        # Prepare test data with proper feature engineering
        try:
            X, y_workout, y_nutrition, df_enhanced = self.model.prepare_training_data(self.df_training)
            test_mask = df_enhanced['split'] == 'test'
            X_test = X[test_mask]
            
            # Scale features for XGBoost
            X_test_xgb = self.model.scaler.transform(X_test)
            xgb_n_pred = self.model.nutrition_model.predict(X_test_xgb)
            
            # Convert back to template IDs
            xgb_n_pred_templates = self.model.nutrition_label_encoder.inverse_transform(xgb_n_pred)
            
            print(f"✅ XGBoost nutrition predictions obtained successfully")
        except Exception as e:
            print(f"⚠️  Error getting XGBoost nutrition predictions: {e}")
            return
        
        # Check RF availability and get predictions
        rf_available = (hasattr(self.model, 'nutrition_rf_model') and 
                       self.model.nutrition_rf_model is not None)
        
        if rf_available:
            try:
                X_test_rf = self.model.rf_scaler.transform(X_test)
                rf_n_pred = self.model.nutrition_rf_model.predict(X_test_rf)
                rf_n_pred_templates = self.model.nutrition_rf_label_encoder.inverse_transform(rf_n_pred)
                print(f"✅ RF nutrition predictions obtained successfully")
            except Exception as e:
                print(f"⚠️  Error getting RF nutrition predictions: {e}")
                rf_available = False
        
        # Calculate metrics for all models
        metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        # Rule-Based metrics
        rule_metrics = [
            accuracy_score(y_n_test, rule_n_pred),
            f1_score(y_n_test, rule_n_pred, average='weighted'),
            precision_score(y_n_test, rule_n_pred, average='weighted'),
            recall_score(y_n_test, rule_n_pred, average='weighted')
        ]
        
        # XGBoost metrics
        xgb_metrics = [
            accuracy_score(y_n_test, xgb_n_pred_templates),
            f1_score(y_n_test, xgb_n_pred_templates, average='weighted'),
            precision_score(y_n_test, xgb_n_pred_templates, average='weighted'),
            recall_score(y_n_test, xgb_n_pred_templates, average='weighted')
        ]
        
        # Create the comparison chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metrics_labels))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, rule_metrics, width, label='Rule-Based', color=self.colors['rule_based'], alpha=0.8)
        ax.bar(x, xgb_metrics, width, label='XGBoost', color=self.colors['xgboost'], alpha=0.8)
        
        if rf_available:
            # Random Forest metrics
            rf_metrics = [
                accuracy_score(y_n_test, rf_n_pred_templates),
                f1_score(y_n_test, rf_n_pred_templates, average='weighted'),
                precision_score(y_n_test, rf_n_pred_templates, average='weighted'),
                recall_score(y_n_test, rf_n_pred_templates, average='weighted')
            ]
            ax.bar(x + width, rf_metrics, width, label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Nutrition Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(rule_metrics):
            ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(xgb_metrics):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if rf_available:
            for i, v in enumerate(rf_metrics):
                ax.text(i + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/22_nutrition_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Nutrition model comparison chart created successfully")
    
    def create_overall_model_performance(self):
        """Overall model performance summary with Rule-Based Baseline"""
        if not self.model.is_trained:
            return
            
        # Get test data for rule-based baseline
        test_data = self.df_training[self.df_training['split'] == 'test']
        if len(test_data) == 0:
            print("⚠️  No test data available for overall performance comparison")
            return
        
        # Get Rule-Based Baseline predictions and calculate accuracy
        try:
            rule_w_pred, rule_n_pred = self.model.predict_rule_based(test_data)
        except AttributeError:
            print("⚠️  Model does not have predict_rule_based method, skipping rule-based comparison")
            return
        
        y_w_test = test_data['workout_template_id'].values
        y_n_test = test_data['nutrition_template_id'].values
        
        rule_scores = [
            accuracy_score(y_w_test, rule_w_pred),
            accuracy_score(y_n_test, rule_n_pred)
        ]
        
        # Get XGBoost info
        xgb_info = self.model.training_info if hasattr(self.model, 'training_info') else {}
        
        # Check RF availability
        rf_available = (hasattr(self.model, 'rf_training_info') and 
                       self.model.rf_training_info is not None and
                       hasattr(self.model, 'workout_rf_model') and 
                       self.model.workout_rf_model is not None)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = ['Workout Models', 'Nutrition Models']
        
        # Get XGBoost scores
        xgb_scores = [
            xgb_info.get('workout_accuracy', 0.0), 
            xgb_info.get('nutrition_accuracy', 0.0)
        ]
        
        x = np.arange(len(models))
        width = 0.25
        
        # Plot bars
        ax.bar(x - width, rule_scores, width, label='Rule-Based', color=self.colors['rule_based'], alpha=0.8)
        ax.bar(x, xgb_scores, width, label='XGBoost', color=self.colors['xgboost'], alpha=0.8)
        
        if rf_available:
            # Get Random Forest scores
            rf_info = self.model.rf_training_info
            rf_scores = [
                rf_info.get('rf_workout_accuracy', 0.0),
                rf_info.get('rf_nutrition_accuracy', 0.0)
            ]
            ax.bar(x + width, rf_scores, width, label='Random Forest', color=self.colors['random_forest'], alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Model Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(rule_scores):
            ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(xgb_scores):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if rf_available:
            for i, v in enumerate(rf_scores):
                ax.text(i + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/23_overall_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Overall model performance created with Rule-Based Baseline")
    
    # Model Comparison Analysis with improved RF detection
    def create_prediction_agreement(self):
        """Model prediction agreement analysis with better RF detection"""
        if not self.model.is_trained:
            return
            
        rf_available = (hasattr(self.model, 'workout_rf_model') and 
                       self.model.workout_rf_model is not None and
                       hasattr(self.model, 'compare_model_predictions'))
        
        if not rf_available:
            print("⚠️  Random Forest models not available for prediction agreement analysis")
            return
            
        try:
            comparison_data = self.model.compare_model_predictions(self.df_training)
        except:
            print("⚠️  Could not get model comparison data")
            return
            
        if not comparison_data:
            print("⚠️  No comparison data available")
            return
            
        # Prediction Agreement Chart
        models = ['Workout Models', 'Nutrition Models']
        workout_diff_pct = comparison_data['workout_differences'] / comparison_data['total_test_samples'] * 100
        nutrition_diff_pct = comparison_data['nutrition_differences'] / comparison_data['total_test_samples'] * 100
        agreement_rates = [100 - workout_diff_pct, 100 - nutrition_diff_pct]
        disagreement_rates = [workout_diff_pct, nutrition_diff_pct]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars1 = ax.bar(x - width/2, agreement_rates, width, 
                      label='Agreement', color=self.colors['success'], alpha=0.8)
        bars2 = ax.bar(x + width/2, disagreement_rates, width, 
                      label='Disagreement', color=self.colors['secondary'], alpha=0.8)
        
        ax.set_title('XGBoost vs Random Forest Prediction Agreement', fontsize=16, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(agreement_rates):
            ax.annotate(f'{v:.1f}%', (i - width/2, v), ha='center', va='bottom', fontweight='bold')
        for i, v in enumerate(disagreement_rates):
            ax.annotate(f'{v:.1f}%', (i + width/2, v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/32_prediction_agreement.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Prediction agreement chart created successfully")
    
    def create_algorithm_diversity(self):
        """Algorithm diversity analysis with better RF detection"""
        if not self.model.is_trained:
            return
            
        rf_available = (hasattr(self.model, 'workout_rf_model') and 
                       self.model.workout_rf_model is not None and
                       hasattr(self.model, 'compare_model_predictions'))
        
        if not rf_available:
            print("⚠️  Random Forest models not available for diversity analysis")
            return
            
        try:
            comparison_data = self.model.compare_model_predictions(self.df_training)
        except:
            print("⚠️  Could not get model comparison data for diversity")
            return
            
        if not comparison_data:
            print("⚠️  No comparison data available for diversity")
            return
            
        workout_diff_pct = comparison_data['workout_differences'] / comparison_data['total_test_samples'] * 100
        nutrition_diff_pct = comparison_data['nutrition_differences'] / comparison_data['total_test_samples'] * 100
        
        diversity_data = {
            'Metric': ['Workout Agreement', 'Nutrition Agreement', 'Overall Diversity'],
            'Value': [100 - workout_diff_pct, 100 - nutrition_diff_pct, (workout_diff_pct + nutrition_diff_pct) / 2]
        }
        
        colors_diversity = [self.colors['primary'], self.colors['accent'], self.colors['success']]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.bar(diversity_data['Metric'], diversity_data['Value'], color=colors_diversity, alpha=0.8)
        ax.set_title('Algorithm Diversity Analysis (XGBoost vs Random Forest)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', (bar.get_x() + bar.get_width() / 2, height), 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/33_algorithm_diversity.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Algorithm diversity chart created successfully")
    

    

    
    # Individual Confusion Matrices
    def create_individual_confusion_matrices(self):
        """Create individual confusion matrices for all models"""
        if not self.model.is_trained:
            return
            
        X, y_workout, y_nutrition, df_enhanced = self.model.prepare_training_data(self.df_training)
        test_mask = df_enhanced['split'] == 'test'
        X_test = X[test_mask]
        y_w_test = y_workout[test_mask]
        y_n_test = y_nutrition[test_mask]
        
        if len(X_test) == 0:
            return
            
        X_test_xgb = self.model.scaler.transform(X_test)
        y_w_test_encoded = self.model.workout_label_encoder.transform(y_w_test)
        y_n_test_encoded = self.model.nutrition_label_encoder.transform(y_n_test)
        xgb_w_pred = self.model.workout_model.predict(X_test_xgb)
        xgb_n_pred = self.model.nutrition_model.predict(X_test_xgb)
        
        rf_available = hasattr(self.model, 'workout_rf_model') and self.model.workout_rf_model
        if rf_available:
            X_test_rf = self.model.rf_scaler.transform(X_test)
            rf_w_pred = self.model.workout_rf_model.predict(X_test_rf)
            rf_n_pred = self.model.nutrition_rf_model.predict(X_test_rf)
        
        # XGBoost Workout Confusion Matrix
        cm_xgb_w = confusion_matrix(y_w_test_encoded, xgb_w_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_xgb_w, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'XGBoost Workout Model Confusion Matrix\nAccuracy: {accuracy_score(y_w_test_encoded, xgb_w_pred):.3f}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Template ID')
        ax.set_ylabel('Actual Template ID')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/24_confusion_xgboost_workout.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # XGBoost Nutrition Confusion Matrix
        cm_xgb_n = confusion_matrix(y_n_test_encoded, xgb_n_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_xgb_n, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'XGBoost Nutrition Model Confusion Matrix\nAccuracy: {accuracy_score(y_n_test_encoded, xgb_n_pred):.3f}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Template ID')
        ax.set_ylabel('Actual Template ID')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/25_confusion_xgboost_nutrition.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if rf_available:
            # Random Forest Workout Confusion Matrix
            cm_rf_w = confusion_matrix(y_w_test_encoded, rf_w_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_rf_w, annot=True, fmt='d', cmap='Oranges', ax=ax)
            ax.set_title(f'Random Forest Workout Model Confusion Matrix\nAccuracy: {accuracy_score(y_w_test_encoded, rf_w_pred):.3f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Template ID')
            ax.set_ylabel('Actual Template ID')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/26_confusion_rf_workout.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Random Forest Nutrition Confusion Matrix
            cm_rf_n = confusion_matrix(y_n_test_encoded, rf_n_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_rf_n, annot=True, fmt='d', cmap='Oranges', ax=ax)
            ax.set_title(f'Random Forest Nutrition Model Confusion Matrix\nAccuracy: {accuracy_score(y_n_test_encoded, rf_n_pred):.3f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Template ID')
            ax.set_ylabel('Actual Template ID')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/27_confusion_rf_nutrition.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # Individual ROC Curves
    def create_individual_roc_curves(self):
        """Create individual ROC curves for all models"""
        if not self.model.is_trained:
            return
            
        X, y_workout, y_nutrition, df_enhanced = self.model.prepare_training_data(self.df_training)
        test_mask = df_enhanced['split'] == 'test'
        X_test = X[test_mask]
        y_w_test = y_workout[test_mask]
        y_n_test = y_nutrition[test_mask]
        
        if len(X_test) == 0:
            return
            
        X_test_xgb = self.model.scaler.transform(X_test)
        y_w_test_encoded = self.model.workout_label_encoder.transform(y_w_test)
        y_n_test_encoded = self.model.nutrition_label_encoder.transform(y_n_test)
        xgb_w_proba = self.model.workout_model.predict_proba(X_test_xgb)
        xgb_n_proba = self.model.nutrition_model.predict_proba(X_test_xgb)
        
        rf_available = hasattr(self.model, 'workout_rf_model') and self.model.workout_rf_model
        if rf_available:
            X_test_rf = self.model.rf_scaler.transform(X_test)
            rf_w_proba = self.model.workout_rf_model.predict_proba(X_test_rf)
            rf_n_proba = self.model.nutrition_rf_model.predict_proba(X_test_rf)
        
        workout_classes = sorted(np.unique(y_w_test_encoded))
        nutrition_classes = sorted(np.unique(y_n_test_encoded))
        
        # XGBoost Workout ROC
        y_w_bin = label_binarize(y_w_test_encoded, classes=workout_classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
        
        if len(workout_classes) > 2:
            for i, color in zip(range(len(workout_classes)), colors):
                if i < y_w_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_w_bin[:, i], xgb_w_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {workout_classes[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('XGBoost Workout Model ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/28_roc_xgboost_workout.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # XGBoost Nutrition ROC
        y_n_bin = label_binarize(y_n_test_encoded, classes=nutrition_classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
        
        if len(nutrition_classes) > 2:
            for i, color in zip(range(len(nutrition_classes)), colors):
                if i < y_n_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_n_bin[:, i], xgb_n_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {nutrition_classes[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('XGBoost Nutrition Model ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/29_roc_xgboost_nutrition.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if rf_available:
            # Random Forest Workout ROC
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
            
            if len(workout_classes) > 2:
                for i, color in zip(range(len(workout_classes)), colors):
                    if i < y_w_bin.shape[1]:
                        fpr, tpr, _ = roc_curve(y_w_bin[:, i], rf_w_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {workout_classes[i]} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Random Forest Workout Model ROC Curves', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/30_roc_rf_workout.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Random Forest Nutrition ROC
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
            
            if len(nutrition_classes) > 2:
                for i, color in zip(range(len(nutrition_classes)), colors):
                    if i < y_n_bin.shape[1]:
                        fpr, tpr, _ = roc_curve(y_n_bin[:, i], rf_n_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {nutrition_classes[i]} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Random Forest Nutrition Model ROC Curves', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/31_roc_rf_nutrition.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    

    
    # Summary Individual Charts
    def create_dataset_summary(self):
        """Dataset summary chart"""
        total_samples = len(self.df_training)
        real_samples = len(self.df_training[self.df_training['data_source'] == 'real'])
        synthetic_samples = len(self.df_training[self.df_training['data_source'] == 'synthetic'])
        
        summary_data = ['Total\nSamples', 'Real\nData', 'Synthetic\nData']
        summary_values = [total_samples, real_samples, synthetic_samples]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.bar(summary_data, summary_values, 
                     color=[self.colors['primary'], self.colors['success'], self.colors['accent']],
                     alpha=0.8)
        ax.set_title('Dataset Summary', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/34_dataset_summary.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_template_coverage(self):
        """Template coverage analysis"""
        workout_templates = len(self.df_training['workout_template_id'].unique())
        nutrition_templates = len(self.df_training['nutrition_template_id'].unique())
        
        template_data = ['Workout\nTemplates', 'Nutrition\nTemplates', 'Total\nTemplates']
        template_values = [workout_templates, nutrition_templates, workout_templates + nutrition_templates]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.bar(template_data, template_values, 
                     color=[self.colors['accent'], self.colors['success'], self.colors['primary']],
                     alpha=0.8)
        ax.set_title('Template Coverage Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Templates')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/35_template_coverage.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_research_findings(self):
        """Key research findings summary with Rule-Based Baseline"""
        total_samples = len(self.df_training)
        real_samples = len(self.df_training[self.df_training['data_source'] == 'real'])
        workout_templates = len(self.df_training['workout_template_id'].unique())
        nutrition_templates = len(self.df_training['nutrition_template_id'].unique())
        
        findings_text = [
            f"✓ {total_samples:,} Total Samples",
            f"✓ {real_samples:,} Real Data Points",
            f"✓ 100% Real Validation/Test",
            f"✓ Natural Distributions Preserved",
            f"✓ {workout_templates} Workout Templates",
            f"✓ {nutrition_templates} Nutrition Templates"
        ]
        
        # Add Rule-Based Baseline performance
        test_data = self.df_training[self.df_training['split'] == 'test']
        if len(test_data) > 0:
            try:
                rule_w_pred, rule_n_pred = self.model.predict_rule_based(test_data)
                y_w_test = test_data['workout_template_id'].values
                y_n_test = test_data['nutrition_template_id'].values
                
                rule_w_acc = accuracy_score(y_w_test, rule_w_pred)
                rule_n_acc = accuracy_score(y_n_test, rule_n_pred)
                
                findings_text.extend([
                    f"✓ Rule-Based Workout: {rule_w_acc:.1%}",
                    f"✓ Rule-Based Nutrition: {rule_n_acc:.1%}"
                ])
            except AttributeError:
                print("⚠️  Model does not have predict_rule_based method, skipping rule-based in research findings")
        
        if self.model.is_trained:
            xgb_info = self.model.training_info
            findings_text.extend([
                f"✓ XGBoost Workout: {xgb_info['workout_accuracy']:.1%}",
                f"✓ XGBoost Nutrition: {xgb_info['nutrition_accuracy']:.1%}"
            ])
            
            if hasattr(self.model, 'rf_training_info') and self.model.rf_training_info:
                rf_info = self.model.rf_training_info
                findings_text.extend([
                    f"✓ RF Workout: {rf_info['rf_workout_accuracy']:.1%}",
                    f"✓ RF Nutrition: {rf_info['rf_nutrition_accuracy']:.1%}"
                ])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.05, 0.95, '\n'.join(findings_text), 
               transform=ax.transAxes,
               fontsize=14, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background'], alpha=0.8))
        ax.set_title('Key Research Findings (with Rule-Based Baseline)', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/36_research_findings.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _list_individual_files(self):
        """List all generated individual files"""
        expected_files = [
            # Dataset Composition (6 files)
            '01_data_source_distribution.png',
            '02_split_distribution.png', 
            '03_fitness_goal_distribution.png',
            '04_activity_level_distribution.png',
            '05_bmi_category_distribution.png',
            '06_gender_distribution.png',
            
            # Data Quality (4 files)
            '07_real_vs_synthetic_by_split.png',
            '08_age_distribution.png',
            '09_bmi_vs_tdee_scatter.png',
            '10_activity_hours_scatter.png',
            
            # Demographics (6 files)
            '11_height_by_gender.png',
            '12_weight_by_gender.png',
            '13_bmi_distribution_with_categories.png',
            '14_bmr_vs_age_by_gender.png',
            '15_tdee_by_activity_level.png',
            '16_age_vs_bmi_by_goal.png',
            
            # Template Analysis (4 files)
            '17_workout_template_distribution.png',
            '18_nutrition_template_distribution.png',
            '19_fat_loss_activity_bmi_heatmap.png',
            '20_top_template_combinations.png',
            
            # Summary (3 files)
            '34_dataset_summary.png',
            '35_template_coverage.png',
            '36_research_findings.png'
        ]
        
        # Add model performance files if models are trained
        if self.model.is_trained:
            model_files = [
                # Model Performance Comparisons (3 files)
                '21_workout_model_comparison.png',
                '22_nutrition_model_comparison.png',
                '23_overall_model_performance.png',
                
                # Confusion Matrices (2-4 files depending on RF availability)
                '24_confusion_xgboost_workout.png',
                '25_confusion_xgboost_nutrition.png',
                
                # ROC Curves (2-4 files depending on RF availability)
                '28_roc_xgboost_workout.png',
                '29_roc_xgboost_nutrition.png'
            ]
            
            rf_available = hasattr(self.model, 'workout_rf_model') and self.model.workout_rf_model
            if rf_available:
                model_files.extend([
                    # Additional RF files
                    '26_confusion_rf_workout.png',
                    '27_confusion_rf_nutrition.png',
                    '30_roc_rf_workout.png',
                    '31_roc_rf_nutrition.png',
                    
                    # Model Comparison Analysis (2 files)
                    '32_prediction_agreement.png',
                    '33_algorithm_diversity.png'
                ])
            
            expected_files.extend(model_files)
        
        # Sort files by number
        expected_files.sort()
        
        generated_files = []
        missing_files = []
        
        for filename in expected_files:
            filepath = os.path.join(self.save_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # Size in KB
                generated_files.append(f"   ✅ {filename} ({file_size:.1f} KB)")
            else:
                missing_files.append(f"   ❌ {filename} (missing)")
        
        print("\n" + "="*80)
        print("INDIVIDUAL CHART GENERATION SUMMARY")
        print("="*80)
        
        print(f"\nGenerated Files ({len(generated_files)}):")
        for file_info in generated_files:
            print(file_info)
        
        if missing_files:
            print(f"\nMissing Files ({len(missing_files)}):")
            for file_info in missing_files:
                print(file_info)
        
        print(f"\nSUMMARY:")
        print(f"   📊 Total charts generated: {len(generated_files)}")
        print(f"   📁 Output directory: {os.path.abspath(self.save_dir)}")
        print(f"   💾 Total size: {sum([os.path.getsize(os.path.join(self.save_dir, f.split()[1])) for f in generated_files if os.path.exists(os.path.join(self.save_dir, f.split()[1]))]) / 1024:.1f} KB")
        
        # Chart categories summary
        categories = {
            "Dataset Composition": (1, 6),
            "Data Quality": (7, 10), 
            "Demographics": (11, 16),
            "Template Analysis": (17, 20),
            "Model Performance": (21, 23) if self.model.is_trained else None,
            "Confusion Matrices": (24, 27) if self.model.is_trained else None,
            "ROC Curves": (28, 31) if self.model.is_trained else None,
            "Model Comparison": (32, 33) if self.model.is_trained and hasattr(self.model, 'workout_rf_model') else None,
            "Summary": (34, 36)
        }
        
        print(f"\nCHART CATEGORIES:")
        for category, range_tuple in categories.items():
            if range_tuple:
                start, end = range_tuple
                count = len([f for f in generated_files if any(f"/{i:02d}_" in f for i in range(start, end + 1))])
                print(f"   📈 {category}: {count} charts")


# Modified runner script integration
def generate_individual_charts(model, df_training, save_dir='visualizations_individual'):
    """
    Main function to generate all individual charts
    
    Args:
        model: Trained XGFitnessAIModel instance
        df_training: Training dataset DataFrame  
        save_dir: Directory to save individual visualizations
    """
    # Create individual visualization suite instance
    viz_suite = XGFitnessIndividualVisualizationSuite(model, df_training, save_dir)
    
    # Generate all individual charts
    viz_suite.generate_all_individual_charts()
    
    return save_dir


# Standalone execution for testing
if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Generate Individual XGFitness AI Charts')
    parser.add_argument('--model', '-m', default='models/xgfitness_ai_model.pkl',
                       help='Path to trained model file')
    parser.add_argument('--data', '-d', default='training_data.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--output', '-o', default='visualizations_individual',
                       help='Output directory for individual charts')
    
    args = parser.parse_args()
    
    print("🎨 XGFitness AI Individual Chart Generator")
    print("="*60)
    
    # Load model
    try:
        print(f"📥 Loading model from: {args.model}")
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a mock model object with the loaded data
        class LoadedModel:
            def __init__(self, model_data):
                for key, value in model_data.items():
                    setattr(self, key, value)
                # Ensure is_trained attribute exists
                self.is_trained = getattr(self, 'is_trained', False) or (
                    hasattr(self, 'workout_model') and getattr(self, 'workout_model', None) is not None
                )
        
        model = LoadedModel(model_data)
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating visualizations with mock model...")
        
        # Create a minimal mock model for demonstration
        class MinimalMockModel:
            def __init__(self):
                self.is_trained = False
                self.training_info = {}
        
        model = MinimalMockModel()
    
    # Load training data
    try:
        print(f"📥 Loading training data from: {args.data}")
        df_training = pd.read_csv(args.data)
        print(f"✅ Training data loaded: {len(df_training)} samples")
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        print("Generating sample data for demonstration...")
        
        # Generate sample data for demonstration
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
        df_training['bmi_category'] = pd.cut(df_training['bmi'], 
                                           bins=[0, 18.5, 25, 30, 100], 
                                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df_training['bmr'] = df_training.apply(lambda row: 
            88.362 + (13.397 * row['weight_kg']) + (4.799 * row['height_cm']) - (5.677 * row['age'])
            if row['gender'] == 'Male' else
            447.593 + (9.247 * row['weight_kg']) + (3.098 * row['height_cm']) - (4.330 * row['age']), 
            axis=1)
        
        activity_multipliers = {'Low Activity': 1.29, 'Moderate Activity': 1.55, 'High Activity': 1.81}
        df_training['tdee'] = df_training['bmr'] * df_training['activity_level'].map(activity_multipliers)
        
        print("✅ Sample data generated successfully")
    
    # Generate individual charts
    try:
        print(f"🎨 Generating individual charts...")
        generate_individual_charts(model, df_training, args.output)
        
        print(f"\n🎉 Individual chart generation completed!")
        print(f"📁 Output directory: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        import traceback
        traceback.print_exc()