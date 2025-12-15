import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib.patches import Patch

class ModelComparisonVisualizer:
    """
    A comprehensive class for visualizing and comparing pre-computed machine learning model results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def _prepare_comparison_data(self, results, X_test, y_test):
        """Prepare data for comparison plots from pre-computed results."""
        comparison_data = []
        
        for model_name, result in results.items():
            if 'best_estimator' in result and result.get('best_score', 0) > 0:
                model = result['best_estimator']
                
                # Ensure model is fitted
                if not hasattr(model, 'classes_'):
                    try:
                        model.fit(X_test, y_test)  # Quick fit if needed
                    except:
                        continue
                
                y_pred = model.predict(X_test)
                
                # Handle ROC AUC calculation carefully
                roc_auc = 0
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    # Check if it's binary or multi-class
                    if y_pred_proba.shape[1] == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:  # Multi-class classification
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except (AttributeError, NotImplementedError, ValueError):
                    # If predict_proba fails or ROC calculation fails, skip it
                    roc_auc = 0
                
                model_metrics = {
                    'Model': model_name.replace('_', ' ').title(),
                    'CV_Score': result.get('best_score', 0),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1_Score': f1_score(y_test, y_pred, average='weighted'),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'ROC_AUC': roc_auc,
                    'Fit_Time': result.get('fit_time', 0),
                    'Best_Params': str(result.get('best_params', {}))
                }
                comparison_data.append(model_metrics)
        
        return pd.DataFrame(comparison_data) if comparison_data else None
    
    def plot_complete_comparison(self, results, X_test, y_test, ascending=True):
        """
        Create comprehensive comparison dashboard from pre-computed results.
        """
        df = self._prepare_comparison_data(results, X_test, y_test)
        if df is None or df.empty:
            print("No valid results to plot. Check if results contain 'best_estimator' and 'best_score'.")
            return None
        
        df_sorted = df.sort_values('CV_Score', ascending=ascending)
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('COMPREHENSIVE MODEL COMPARISON DASHBOARD', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_sorted)))
        
        # 1. Main Performance Chart
        ax1 = fig.add_subplot(gs[0, :2])
        bars = ax1.barh(df_sorted['Model'], df_sorted['CV_Score'], color=colors)
        ax1.set_xlabel('CV F1 Score', fontweight='bold')
        ax1.set_title('Model Performance Ranking (Cross-Validation)', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Multiple Metrics Radar Chart (exclude ROC_AUC if all zeros)
        ax2 = fig.add_subplot(gs[0, 2], polar=True)
        
        # Check if ROC_AUC has valid values
        metrics_to_plot = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
        if df_sorted['ROC_AUC'].sum() > 0:
            metrics_to_plot.append('ROC_AUC')
        
        angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            values = row[metrics_to_plot].tolist()
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
            ax2.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_to_plot)
        ax2.set_yticklabels([])
        ax2.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
        
        # 3. CV Score vs Test Performance
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(df_sorted['CV_Score'], df_sorted['F1_Score'], 
                             s=200, c=range(len(df_sorted)), cmap='viridis', alpha=0.7)
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax3.annotate(row['Model'].split()[0], 
                        (row['CV_Score'], row['F1_Score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect correlation')
        ax3.set_xlabel('CV Score (Training)')
        ax3.set_ylabel('Test F1 Score')
        ax3.set_title('CV vs Test Performance', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Training Time Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        if df_sorted['Fit_Time'].sum() > 0:  # Only plot if timing data exists
            time_bars = ax4.barh(df_sorted['Model'], df_sorted['Fit_Time'], color=colors)
            ax4.set_xlabel('Training Time (seconds)')
            ax4.set_title('Model Training Time', fontsize=14, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            
            for bar in time_bars:
                width = bar.get_width()
                ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}s', ha='left', va='center', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training Time (Data Not Available)', fontsize=14, fontweight='bold')
        
        # 5. Metrics Heatmap (exclude ROC_AUC if all zeros)
        ax5 = fig.add_subplot(gs[1, 2])
        heatmap_metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
        if df_sorted['ROC_AUC'].sum() > 0:
            heatmap_metrics.append('ROC_AUC')
            
        metrics_df = df_sorted[heatmap_metrics]
        sns.heatmap(metrics_df.set_index(df_sorted['Model']), 
                    annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Score'})
        ax5.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        
        # 6. Parameter Complexity
        ax6 = fig.add_subplot(gs[2, :])
        param_complexity = []
        for params in df_sorted['Best_Params']:
            complexity = len(str(params).split(',')) if params != 'nan' and params != '{}' else 0
            param_complexity.append(complexity)
        
        bars = ax6.bar(df_sorted['Model'], param_complexity, color=colors)
        ax6.set_ylabel('Number of Tuned Parameters')
        ax6.set_title('Model Complexity (Number of Tuned Hyperparameters)', 
                     fontsize=14, fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, complexity in zip(bars, param_complexity):
            if complexity > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., complexity + 0.1,
                        f'{complexity}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return df_sorted
    
    def plot_individual_scorecards(self, results, X_test, y_test):
        """
        Create individual model scorecards from pre-computed results.
        """
        df = self._prepare_comparison_data(results, X_test, y_test)
        if df is None:
            print("No valid results to plot.")
            return
        
        n_models = len(df)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(axes):
                break
                
            model_name = row['Model']
            metrics = [row['Accuracy'], row['F1_Score'], row['Precision'], row['Recall']]
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            
            bars = axes[i].bar(metric_names, metrics, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            
            for bar, metric in zip(bars, metrics):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{metric:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'{model_name}\nCV Score: {row["CV_Score"]:.3f}', 
                            fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_trends(self, results, X_test, y_test, ascending=True):
        """
        Show performance trends across different metrics from pre-computed results.
        """
        df = self._prepare_comparison_data(results, X_test, y_test)
        if df is None:
            print("No valid results to plot.")
            return
        
        # Add model type information
        model_types = []
        for model_name in df['Model']:
            lower_name = model_name.lower()
            if any(x in lower_name for x in ['tree', 'forest', 'boosting', 'gradient']):
                model_types.append('Tree-Based')
            elif any(x in lower_name for x in ['logistic', 'regression', 'linear', 'svm']):
                model_types.append('Linear')
            else:
                model_types.append('Other')
        
        df['Model_Type'] = model_types
        df_sorted = df.sort_values('CV_Score', ascending=ascending)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by model type
        color_map = {'Tree-Based': '#FF6B6B', 'Linear': '#4ECDC4', 'Other': '#45B7D1'}
        colors = [color_map[typ] for typ in df_sorted['Model_Type']]
        
        bars = ax.barh(df_sorted['Model'], df_sorted['CV_Score'], color=colors)
        ax.set_xlabel('CV F1 Score', fontweight='bold')
        ax.set_title('Model Performance Ranking', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels and model type indicators
        for bar, score, model_type in zip(bars, df_sorted['CV_Score'], df_sorted['Model_Type']):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
            
            # Add model type indicator
            ax.text(0.02, bar.get_y() + bar.get_height()/2, 
                   model_type, ha='left', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Add legend
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Tree-Based Models'),
            Patch(facecolor='#4ECDC4', label='Linear Models'),
            Patch(facecolor='#45B7D1', label='Other Models')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        return df_sorted
    
    def create_ranking_table(self, results, X_test, y_test):
        """
        Create a ranking table from pre-computed results.
        """
        df = self._prepare_comparison_data(results, X_test, y_test)
        if df is None:
            return None
        
        ranking_df = df.copy()
        ranking_df = ranking_df.sort_values('CV_Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df[['Rank', 'Model', 'CV_Score', 'Accuracy', 'F1_Score', 
                          'Precision', 'Recall', 'Fit_Time']]
    
    def print_ranking_table(self, results, X_test, y_test):
        """
        Print formatted ranking table from pre-computed results.
        """
        ranking_df = self.create_ranking_table(results, X_test, y_test)
        if ranking_df is None:
            print("No results available.")
            return
        
        print("🏆 FINAL MODEL RANKING 🏆")
        print("=" * 80)
        
        # Format the table
        formatted_df = ranking_df.copy()
        for col in ['CV_Score', 'Accuracy', 'F1_Score', 'Precision', 'Recall']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.4f}')
        
        # Format time if available
        if formatted_df['Fit_Time'].sum() > 0:
            formatted_df['Fit_Time'] = formatted_df['Fit_Time'].apply(lambda x: f'{x:.2f}s')
        else:
            formatted_df['Fit_Time'] = 'N/A'
        
        print(formatted_df.to_string(index=False))
        print("=" * 80)
    
    def run_complete_analysis(self, results, X_test, y_test, ascending=True):
        """
        Run complete analysis pipeline on pre-computed results.
        """
        print("📊 Starting model comparison analysis...")
        
        # Check if results are valid
        if not results or not any('best_estimator' in r for r in results.values()):
            print("❌ No valid results found. Ensure results contain 'best_estimator' keys.")
            return None
        
        print("\n1. 📈 Generating comprehensive comparison dashboard...")
        comparison_df = self.plot_complete_comparison(results, X_test, y_test, ascending)
        
        print("\n2. 📊 Creating individual model scorecards...")
        self.plot_individual_scorecards(results, X_test, y_test)
        
        print("\n3. 📋 Generating performance trends...")
        self.plot_performance_trends(results, X_test, y_test, ascending)
        
        print("\n4. 🏆 Displaying ranking table...")
        self.print_ranking_table(results, X_test, y_test)
        
        print("\n✅ Analysis complete!")
        
        return comparison_df