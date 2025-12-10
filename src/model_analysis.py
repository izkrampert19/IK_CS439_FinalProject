"""

Python script that generates a CSV showing the prediction data for 
each demographic, and a PNG with 4 plots comparing the performance by both models.

Run this file after running model_training.py -- it relies on previously
trained models to construct the fairness_analysis CSV and the model_comparison.png

"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# --- Loading in prev. saved models and data ---
def load_models_and_data(load_dir='../models/'):

    with open(f'{load_dir}logistic_regression_model.pkl', 'rb') as f:
        logreg_model = pickle.load(f)
    
    with open(f'{load_dir}naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    with open(f'{load_dir}training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)
    
    return (logreg_model, nb_model, 
            training_data['X_train'], training_data['X_test'],
            training_data['y_train'], training_data['y_test'],
            training_data['label_encoders'], training_data['feature_names'],
            training_data['df_original'])

# --- Model evaluation ---
# Store everything into a results set, to be returned and used in plot comparison
def evaluate_models(logreg_model, nb_model, X_test, y_test):

    results = {}
    
    for model_name, model in [('Logistic Regression', logreg_model), ('Naive Bayes', nb_model)]:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        con_mat = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': con_mat
        }
        
        # Printing out results in terminal
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return results

# --- Creating PNG with plots ---
# shows ROC Curves, Performance Metrics, and Confusion Matrices for both Models
def compare_plots(results, y_test, save_path='../results/model_comparison.png'):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Traffic Stop Prediction Model Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    logreg_scores = [results['Logistic Regression'][m] for m in metrics]
    nb_scores = [results['Naive Bayes'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, logreg_scores, width, label='Logistic Regression', alpha=0.8, color='#2E86AB')
    axes[0, 0].bar(x + width/2, nb_scores, width, label='Naive Bayes', alpha=0.8, color='#A23B72')
    axes[0, 0].set_xlabel('Metrics', fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontweight='bold')
    axes[0, 0].set_title('Performance Metrics Comparison', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    
    for model_name, color in [('Logistic Regression', '#2E86AB'), ('Naive Bayes', '#A23B72')]:
        fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
        auc = results[model_name]['roc_auc']
        axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2.5, color=color)
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
    axes[0, 1].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0, 1].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0, 1].set_title('ROC Curves', fontweight='bold')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    sns.heatmap(results['Logistic Regression']['confusion_matrix'], 
               annot=True, fmt=',', cmap='Blues', ax=axes[1, 0],
               xticklabels=['Not Stopped', 'Stopped'],
               yticklabels=['Not Stopped', 'Stopped'],
               cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
    axes[1, 0].set_ylabel('True Label', fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label', fontweight='bold')
    
    sns.heatmap(results['Naive Bayes']['confusion_matrix'], 
               annot=True, fmt=',', cmap='Purples', ax=axes[1, 1],
               xticklabels=['Not Stopped', 'Stopped'],
               yticklabels=['Not Stopped', 'Stopped'],
               cbar_kws={'label': 'Count'})
    axes[1, 1].set_title('Confusion Matrix - Naive Bayes', fontweight='bold')
    axes[1, 1].set_ylabel('True Label', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -- Analyze features of the logistic regression model
def feature_analysis(logreg_model, feature_names):
    coefficients = logreg_model.coef_[0]
    
    print(f"\n{'Feature':<20} {'Coefficient':<12} {'Odds Ratio':<12}")
    print("-" * 50)
    
    for i, feature in enumerate(feature_names):
        coef = coefficients[i]
        odds_ratio = np.exp(coef)
        print(f"{feature:<20} {coef:>11.4f} {odds_ratio:>11.4f}")

# --- Generate the fairness analysis CSV ---
# To use as a visual when looking at demographic prediction data
def fairness_analysis(logreg_model, nb_model, df_original, feature_names, label_encoders, save_path='../results/fairness_analysis.csv'):
    
    X_full = df_original[feature_names].copy()
    for col in feature_names:
        X_full[col] = label_encoders[col].transform(df_original[col])
    
    df_analysis = df_original.copy()
    df_analysis['LR_Prediction'] = logreg_model.predict(X_full.values)
    df_analysis['LR_Probability'] = logreg_model.predict_proba(X_full.values)[:, 1]
    df_analysis['NB_Prediction'] = nb_model.predict(X_full.values)
    
    race_analysis = df_analysis.groupby('Driver_Race').agg({
        'LR_Prediction': ['mean', 'count'],
        'Stopped': 'mean',
        'LR_Probability': 'mean'
    }).round(4)
    race_analysis.columns = ['Predicted_Stop_Rate', 'Count', 'Actual_Stop_Rate', 'Avg_Probability']
    
    age_analysis = df_analysis.groupby('Driver_Age_Group').agg({
        'LR_Prediction': ['mean', 'count'],
        'Stopped': 'mean',
        'LR_Probability': 'mean'
    }).round(4)
    age_analysis.columns = ['Predicted_Stop_Rate', 'Count', 'Actual_Stop_Rate', 'Avg_Probability']
    
    sex_analysis = df_analysis.groupby('Driver_Sex').agg({
        'LR_Prediction': ['mean', 'count'],
        'Stopped': 'mean',
        'LR_Probability': 'mean'
    }).round(4)
    sex_analysis.columns = ['Predicted_Stop_Rate', 'Count', 'Actual_Stop_Rate', 'Avg_Probability']
    
    fairness_report = pd.concat([
        race_analysis.assign(Demographic='Race').reset_index().rename(columns={'Driver_Race': 'Category'}),
        age_analysis.assign(Demographic='Age_Group').reset_index().rename(columns={'Driver_Age_Group': 'Category'}),
        sex_analysis.assign(Demographic='Sex').reset_index().rename(columns={'Driver_Sex': 'Category'})
    ])
    
    fairness_report.to_csv(save_path, index=False)
    
    print("\nBy Race:")
    print(race_analysis)
    print("\nBy Age Group:")
    print(age_analysis)
    print("\nBy Sex:")
    print(sex_analysis)

# --- Main execution ---
if __name__ == "__main__":

    # Loading in the models and data
    (logreg_model, nb_model, X_train, X_test, y_train, y_test, 
     label_encoders, feature_names, df_original) = load_models_and_data()
    
    # obtaining results
    results = evaluate_models(logreg_model, nb_model, X_test, y_test)
    
    # Comparing model performance, and printing the best one in terminal
    logreg_auc = results['Logistic Regression']['roc_auc']
    nb_auc = results['Naive Bayes']['roc_auc']
    best_model = 'Logistic Regression' if logreg_auc > nb_auc else 'Naive Bayes' if nb_auc > logreg_auc else 'Tie'
    print(f"\nBest performing model: {best_model}")
    
    # Generating data visuals + printing results
    compare_plots(results, y_test)
    feature_analysis(logreg_model, feature_names)
    fairness_analysis(logreg_model, nb_model, df_original, feature_names, label_encoders)