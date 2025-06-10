#!/usr/bin/env python3
"""
Comprehensive Model Performance Visualization
Shows training curves, confusion matrix, per-class metrics, and more!
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_training_history(results_path):
    """Load training history from YAML file."""
    history_file = Path(results_path) / "models" / "training_history.yaml"
    with open(history_file, 'r') as f:
        history = yaml.safe_load(f)
    return history

def load_test_results(results_path):
    """Load test results from YAML file."""
    results_file = Path(results_path) / "models" / "test_results.yaml"
    with open(results_file, 'r') as f:
        results = yaml.safe_load(f)
    return results

def plot_training_curves(history, save_path):
    """Create comprehensive training curves visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training & Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training & Validation Accuracy
    axes[0, 1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate Schedule (if available)
    axes[1, 0].plot(epochs, history['train_loss'], 'g-', linewidth=2)
    axes[1, 0].set_title('Loss Reduction Progress', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting Analysis
    gap = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Analysis (Val - Train Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def create_performance_summary(history, test_results, save_path):
    """Create a comprehensive performance summary."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Final Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        test_results['metrics']['accuracy'] * 100,
        test_results['metrics']['precision'] * 100,
        test_results['metrics']['recall'] * 100,
        test_results['metrics']['f1_score'] * 100
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.8)
    axes[0, 0].set_title('Final Test Performance', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score (%)')
    axes[0, 0].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Training Progress
    best_epoch = np.argmax(history['val_acc'])
    axes[0, 1].plot(history['train_acc'], label='Training', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 1].set_title('Accuracy Progress', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Improvement Analysis
    initial_acc = history['val_acc'][0] * 100
    final_acc = max(history['val_acc']) * 100
    improvement = final_acc - initial_acc
    
    improvement_data = ['Initial Validation', 'Final Validation', 'Improvement']
    improvement_values = [initial_acc, final_acc, improvement]
    improvement_colors = ['#FF9999', '#66B2FF', '#90EE90']
    
    bars = axes[0, 2].bar(improvement_data, improvement_values, color=improvement_colors, alpha=0.8)
    axes[0, 2].set_title('Learning Progress', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Accuracy (%)')
    
    for bar, value in zip(bars, improvement_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Training Statistics
    total_epochs = len(history['train_loss'])
    best_val_acc = max(history['val_acc']) * 100
    final_train_acc = history['train_acc'][-1] * 100
    
    stats_text = f"""
    📊 TRAINING STATISTICS
    {'='*25}
    Total Epochs: {total_epochs}
    Best Validation Accuracy: {best_val_acc:.2f}%
    Final Training Accuracy: {final_train_acc:.2f}%
    
    📈 TEST RESULTS
    {'='*25}
    Accuracy: {test_results['metrics']['accuracy']*100:.2f}%
    Precision: {test_results['metrics']['precision']*100:.2f}%
    Recall: {test_results['metrics']['recall']*100:.2f}%
    F1-Score: {test_results['metrics']['f1_score']*100:.2f}%
    
    🎯 MODEL PERFORMANCE
    ={'='*25}
    Classification Quality: {"Excellent" if test_results['metrics']['accuracy'] > 0.8 else "Good" if test_results['metrics']['accuracy'] > 0.6 else "Fair" if test_results['metrics']['accuracy'] > 0.4 else "Needs Improvement"}
    Balanced Performance: {"Yes" if abs(test_results['metrics']['precision'] - test_results['metrics']['recall']) < 0.05 else "No"}
    """
    
    axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Loss Comparison
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    loss_data = ['Training Loss', 'Validation Loss']
    loss_values = [final_train_loss, final_val_loss]
    
    axes[1, 1].bar(loss_data, loss_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[1, 1].set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    
    # Convergence Analysis
    axes[1, 2].plot(history['val_loss'], color='red', linewidth=2, label='Validation Loss')
    axes[1, 2].axhline(y=min(history['val_loss']), color='green', linestyle='--', alpha=0.7, label='Best Loss')
    axes[1, 2].set_title('Loss Convergence', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison(history, save_path):
    """Create model comparison and baseline analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Baseline vs Model Performance
    random_accuracy = 100 / 31  # 31 classes
    final_accuracy = max(history['val_acc']) * 100
    
    comparison_data = ['Random Baseline', 'Our Model', 'Improvement']
    comparison_values = [random_accuracy, final_accuracy, final_accuracy - random_accuracy]
    comparison_colors = ['#FF6B6B', '#4ECDC4', '#90EE90']
    
    bars = axes[0].bar(comparison_data, comparison_values, color=comparison_colors, alpha=0.8)
    axes[0].set_title('Model vs Baseline Performance', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    
    for bar, value in zip(bars, comparison_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Learning Speed Analysis
    epochs_to_50_percent = None
    for i, acc in enumerate(history['val_acc']):
        if acc >= 0.5:
            epochs_to_50_percent = i + 1
            break
    
    milestones = ['20% Acc', '30% Acc', '40% Acc', '50% Acc', 'Best Acc']
    milestone_epochs = []
    
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        for i, acc in enumerate(history['val_acc']):
            if acc >= threshold:
                milestone_epochs.append(i + 1)
                break
        else:
            milestone_epochs.append(len(history['val_acc']))
    
    milestone_epochs.append(np.argmax(history['val_acc']) + 1)
    
    axes[1].plot(milestone_epochs, milestones, marker='o', linewidth=2, markersize=8)
    axes[1].set_title('Learning Speed Analysis', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy Milestone')
    axes[1].grid(True, alpha=0.3)
    
    # Training Efficiency
    total_improvement = (max(history['val_acc']) - history['val_acc'][0]) * 100
    epochs_used = len(history['val_acc'])
    efficiency = total_improvement / epochs_used
    
    efficiency_text = f"""
    🚀 TRAINING EFFICIENCY
    ={'='*25}
    Total Improvement: {total_improvement:.2f}%
    Epochs Used: {epochs_used}
    Improvement per Epoch: {efficiency:.2f}%
    
    Time to 50% Accuracy: {epochs_to_50_percent if epochs_to_50_percent else 'Not reached'} epochs
    
    📊 PERFORMANCE RATING
    ={'='*25}
    Speed: {"Fast" if epochs_to_50_percent and epochs_to_50_percent <= 10 else "Moderate" if epochs_to_50_percent and epochs_to_50_percent <= 15 else "Slow"}
    Stability: {"High" if np.std(history['val_acc'][-5:]) < 0.01 else "Moderate"}
    Final Performance: {"Excellent" if max(history['val_acc']) > 0.8 else "Good" if max(history['val_acc']) > 0.6 else "Fair"}
    """
    
    axes[2].text(0.05, 0.95, efficiency_text, transform=axes[2].transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all visualizations."""
    print("🎨 GENERATING PERFORMANCE VISUALIZATIONS")
    print("=" * 50)
    
    # Paths
    results_path = Path("experiments/real_fish_classifier")
    save_path = Path("visualizations")
    save_path.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading training history and test results...")
    history = load_training_history(results_path)
    test_results = load_test_results(results_path)
    
    print(f"✅ Loaded {len(history['train_loss'])} epochs of training data")
    print(f"🎯 Final Test Accuracy: {test_results['metrics']['accuracy']*100:.2f}%")
    
    # Generate visualizations
    print("\n🎨 Creating visualizations...")
    
    print("   📈 Training curves...")
    plot_training_curves(history, save_path)
    
    print("   📊 Performance summary...")
    create_performance_summary(history, test_results, save_path)
    
    print("   🔍 Model comparison...")
    create_model_comparison(history, save_path)
    
    print(f"\n✅ All visualizations saved to: {save_path}")
    print("\n🎉 VISUALIZATION COMPLETE!")
    
    # Print summary
    print("\n" + "="*60)
    print("🏆 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"🎯 Test Accuracy:  {test_results['metrics']['accuracy']*100:.2f}%")
    print(f"🎯 Test Precision: {test_results['metrics']['precision']*100:.2f}%")
    print(f"🎯 Test Recall:    {test_results['metrics']['recall']*100:.2f}%")
    print(f"🎯 Test F1-Score:  {test_results['metrics']['f1_score']*100:.2f}%")
    print(f"📈 Best Val Acc:   {max(history['val_acc'])*100:.2f}%")
    print(f"📉 Final Loss:     {history['val_loss'][-1]:.3f}")
    print(f"⚡ Total Epochs:   {len(history['train_loss'])}")
    
    improvement = (max(history['val_acc']) - history['val_acc'][0]) * 100
    print(f"🚀 Improvement:    {improvement:.2f}% over training")
    
    vs_random = (test_results['metrics']['accuracy'] * 100) - (100/31)
    print(f"🎲 vs Random:      +{vs_random:.1f}% better than random guessing")

if __name__ == "__main__":
    main() 