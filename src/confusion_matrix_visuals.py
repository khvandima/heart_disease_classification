import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class complete_cm_analysis:
    @staticmethod  
    def plot_comparable_confusion_matrices(results, X_train, y_train, X_test, y_test):
        n_models = sum(1 for result in results.values() if 'best_estimator' in result)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))
        
        # Handle single model case
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= n_models or 'best_estimator' not in result:
                continue
                
            model = result['best_estimator']
            
            # Train the model if not already trained
            if not hasattr(model, 'classes_'):
                model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred)
            cm_test = confusion_matrix(y_test, y_test_pred)
            
            # Get class labels
            classes = unique_labels(y_train, y_test)
            
            # Plot training confusion matrix
            im1 = axes[i, 0].imshow(cm_train, interpolation='nearest', cmap='Blues')
            axes[i, 0].set_title(f'{model_name.replace("_", " ").title()}\nTraining Set', 
                               fontweight='bold', fontsize=12)
            axes[i, 0].set_xlabel('Predicted')
            axes[i, 0].set_ylabel('Actual')
            
            # Plot test confusion matrix
            im2 = axes[i, 1].imshow(cm_test, interpolation='nearest', cmap='Blues')
            axes[i, 1].set_title(f'{model_name.replace("_", " ").title()}\nTest Set', 
                               fontweight='bold', fontsize=12)
            axes[i, 1].set_xlabel('Predicted')
            axes[i, 1].set_ylabel('Actual')
            
            # Add values to cells
            for (j, k), val in np.ndenumerate(cm_train):
                axes[i, 0].text(k, j, f'{val}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > cm_train.max()/2 else 'black')
            
            for (j, k), val in np.ndenumerate(cm_test):
                axes[i, 1].text(k, j, f'{val}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > cm_test.max()/2 else 'black')
            
            # Set tick labels
            for ax in [axes[i, 0], axes[i, 1]]:
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[i, 0])
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        plt.show()

    @staticmethod  # ADD STATIC METHOD DECORATOR
    def plot_confusion_matrix_differences(results, X_train, y_train, X_test, y_test):
    
        n_models = sum(1 for result in results.values() if 'best_estimator' in result)
        fig, axes = plt.subplots(n_models, 3, figsize=(20, 5 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= n_models or 'best_estimator' not in result:
                continue
                
            model = result['best_estimator']
            # Check if already fitted to avoid refitting
            if not hasattr(model, 'classes_'):
                model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred, normalize='true')
            cm_test = confusion_matrix(y_test, y_test_pred, normalize='true')
            cm_diff = cm_train - cm_test  # Positive = overfitting on training
            
            classes = unique_labels(y_train, y_test)
            
            # Plot training confusion matrix
            im1 = axes[i, 0].imshow(cm_train, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            axes[i, 0].set_title(f'{model_name.replace("_", " ").title()}\nTraining Set', 
                               fontweight='bold', fontsize=10)
            axes[i, 0].set_xlabel('Predicted')
            axes[i, 0].set_ylabel('Actual')
            
            # Plot test confusion matrix
            im2 = axes[i, 1].imshow(cm_test, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            axes[i, 1].set_title(f'{model_name.replace("_", " ").title()}\nTest Set', 
                               fontweight='bold', fontsize=10)
            axes[i, 1].set_xlabel('Predicted')
            axes[i, 1].set_ylabel('Actual')
            
            # Plot difference matrix (overfitting analysis)
            im3 = axes[i, 2].imshow(cm_diff, interpolation='nearest', cmap='RdBu_r', 
                                   vmin=-1, vmax=1)
            axes[i, 2].set_title(f'{model_name.replace("_", " ").title()}\nTraining - Test Difference\n(Positive = Overfitting)', 
                               fontweight='bold', fontsize=10)
            axes[i, 2].set_xlabel('Predicted')
            axes[i, 2].set_ylabel('Actual')
            
            # Add values to cells
            for (j, k), val in np.ndenumerate(cm_train):
                axes[i, 0].text(k, j, f'{val:.2f}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > 0.5 else 'black')
            
            for (j, k), val in np.ndenumerate(cm_test):
                axes[i, 1].text(k, j, f'{val:.2f}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > 0.5 else 'black')
            
            for (j, k), val in np.ndenumerate(cm_diff):
                axes[i, 2].text(k, j, f'{val:+.2f}', ha='center', va='center', 
                              fontweight='bold', color='white' if abs(val) > 0.25 else 'black')
            
            # Set tick labels
            for j in range(3):
                axes[i, j].set_xticks(range(len(classes)))
                axes[i, j].set_yticks(range(len(classes)))
                axes[i, j].set_xticklabels(classes)
                axes[i, j].set_yticklabels(classes)
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[i, 0], label='Accuracy')
            plt.colorbar(im2, ax=axes[i, 1], label='Accuracy')
            plt.colorbar(im3, ax=axes[i, 2], label='Difference')
        
        plt.tight_layout()
        plt.show()

    @staticmethod  # ADD STATIC METHOD DECORATOR
    def plot_normalized_confusion_matrices(results, X_train, y_train, X_test, y_test):
    
        n_models = sum(1 for result in results.values() if 'best_estimator' in result)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= n_models or 'best_estimator' not in result:
                continue
                
            model = result['best_estimator']
            # Check if already fitted to avoid refitting
            if not hasattr(model, 'classes_'):
                model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Normalized confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred, normalize='true')
            cm_test = confusion_matrix(y_test, y_test_pred, normalize='true')
            
            classes = unique_labels(y_train, y_test)
            
            # Plot training confusion matrix
            im1 = axes[i, 0].imshow(cm_train, interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
            axes[i, 0].set_title(f'{model_name.replace("_", " ").title()}\nTraining Set (Normalized)', 
                               fontweight='bold', fontsize=12)
            axes[i, 0].set_xlabel('Predicted')
            axes[i, 0].set_ylabel('Actual')
            
            # Plot test confusion matrix
            im2 = axes[i, 1].imshow(cm_test, interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title(f'{model_name.replace("_", " ").title()}\nTest Set (Normalized)', 
                               fontweight='bold', fontsize=12)
            axes[i, 1].set_xlabel('Predicted')
            axes[i, 1].set_ylabel('Actual')
            
            # Add values to cells
            for (j, k), val in np.ndenumerate(cm_train):
                axes[i, 0].text(k, j, f'{val:.2f}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > 0.5 else 'black')
            
            for (j, k), val in np.ndenumerate(cm_test):
                axes[i, 1].text(k, j, f'{val:.2f}', ha='center', va='center', 
                              fontweight='bold', color='white' if val > 0.5 else 'black')
            
            # Set tick labels
            for ax in [axes[i, 0], axes[i, 1]]:
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[i, 0], label='Percentage')
            plt.colorbar(im2, ax=axes[i, 1], label='Percentage')
        
        plt.tight_layout()
        plt.show()

    @staticmethod  # ADD STATIC METHOD DECORATOR
    def plot_confusion_matrix_stats(results, X_train, y_train, X_test, y_test):
    
        stats_data = []
        
        for model_name, result in results.items():
            if 'best_estimator' not in result:
                continue
                
            model = result['best_estimator']
            # Check if already fitted to avoid refitting
            if not hasattr(model, 'classes_'):
                model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred)
            cm_test = confusion_matrix(y_test, y_test_pred)
            
            # Calculate metrics
            train_accuracy = np.trace(cm_train) / np.sum(cm_train)
            test_accuracy = np.trace(cm_test) / np.sum(cm_test)
            
            # Class-wise precision and recall
            train_precision = np.diag(cm_train) / np.sum(cm_train, axis=0)
            train_recall = np.diag(cm_train) / np.sum(cm_train, axis=1)
            test_precision = np.diag(cm_test) / np.sum(cm_test, axis=0)
            test_recall = np.diag(cm_test) / np.sum(cm_test, axis=1)
            
            stats_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train_Accuracy': train_accuracy,
                'Test_Accuracy': test_accuracy,
                'Train_Precision_Mean': np.mean(train_precision),
                'Test_Precision_Mean': np.mean(test_precision),
                'Train_Recall_Mean': np.mean(train_recall),
                'Test_Recall_Mean': np.mean(test_recall),
                'Overfitting_Score': train_accuracy - test_accuracy
            })
        
        if not stats_data:
            return
        
        df = pd.DataFrame(stats_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        x = np.arange(len(df))
        width = 0.35
        axes[0, 0].bar(x - width/2, df['Train_Accuracy'], width, label='Train', alpha=0.8)
        axes[0, 0].bar(x + width/2, df['Test_Accuracy'], width, label='Test', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training vs Test Accuracy', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(df['Model'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Overfitting score
        axes[0, 1].bar(df['Model'], df['Overfitting_Score'], color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Overfitting Score (Train - Test)')
        axes[0, 1].set_title('Overfitting Analysis', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(alpha=0.3)
        
        # Precision comparison
        axes[1, 0].bar(x - width/2, df['Train_Precision_Mean'], width, label='Train', alpha=0.8)
        axes[1, 0].bar(x + width/2, df['Test_Precision_Mean'], width, label='Test', alpha=0.8)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Mean Precision')
        axes[1, 0].set_title('Training vs Test Precision', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df['Model'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Recall comparison
        axes[1, 1].bar(x - width/2, df['Train_Recall_Mean'], width, label='Train', alpha=0.8)
        axes[1, 1].bar(x + width/2, df['Test_Recall_Mean'], width, label='Test', alpha=0.8)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Mean Recall')
        axes[1, 1].set_title('Training vs Test Recall', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df['Model'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df

    @staticmethod  # ADD STATIC METHOD DECORATOR
    def run_complete_cm_analysis(results, X_train, y_train, X_test, y_test):
  
        print("🔍 Starting comprehensive confusion matrix analysis...")
        
        print("\n1. 📊 Side-by-Side Confusion Matrices")
        complete_cm_analysis.plot_comparable_confusion_matrices(results, X_train, y_train, X_test, y_test)
        
        print("\n2. 📈 Normalized Confusion Matrices")
        complete_cm_analysis.plot_normalized_confusion_matrices(results, X_train, y_train, X_test, y_test)
        
        print("\n3. ⚖️  Overfitting Analysis (Difference Heatmaps)")
        complete_cm_analysis.plot_confusion_matrix_differences(results, X_train, y_train, X_test, y_test)
        
        print("\n4. 📋 Statistical Summary")
        stats_df = complete_cm_analysis.plot_confusion_matrix_stats(results, X_train, y_train, X_test, y_test)
        
        if stats_df is not None:
            print("\n📊 Performance Statistics Summary:")
            print("=" * 80)
            print(stats_df.round(4).to_string(index=False))
            print("=" * 80)
        
        print("\n✅ Confusion matrix analysis completed!")