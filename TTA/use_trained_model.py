import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from opportunity_model import TradingOpportunityModel
from CronusV1.TTA.enhanced_data_loader_v1 import prepare_enhanced_trading_data

def load_best_model(models_dir):
    """Load the best model from the models directory"""
    model_files = [f for f in os.listdir(models_dir) if f.startswith('model_epoch_')]
    if not model_files:
        raise ValueError(f"No model files found in {models_dir}")
    
    # Sort by epoch number and timestamp to get the latest
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"Loading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, metadata = TradingOpportunityModel.load_model(model_path, device)
    
    return model, metadata

def predict_trends(model, data_path, max_samples=None, output_dir=None):
    """Make predictions using the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small batch size for testing
    batch_size = 32
    
    # Prepare data (using a small training set for standardization)
    _, _, test_loader, data_metadata = prepare_enhanced_trading_data(
        csv_path=data_path,
        batch_size=batch_size,
        max_samples=max_samples,
        train_ratio=0.1,  # Use a small portion for training to allow the scaler to fit
        val_ratio=0.0,
        test_ratio=0.9,  # Use most data for testing
        window_size=50,
        visualize=False
    )
    
    print(f"Prepared {data_metadata['test_size']} samples for prediction")
    
    # Make predictions
    print("Making predictions...")
    model.eval()
    
    all_predictions = []
    all_trends = []
    all_directions = []
    all_confidences = []
    timestamps = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch
            features = batch['features'].to(device)
            trend = batch['trend_strength'].to(device)
            regimes = batch['regimes'].to(device) if 'regimes' in batch and batch['regimes'] is not None else None
            subregimes = batch['subregimes'].to(device) if 'subregimes' in batch and batch['subregimes'] is not None else None
            time_features = batch['time_features'].to(device) if 'time_features' in batch and batch['time_features'] is not None else None
            
            # Get predictions
            predictions = model.predict(
                features, regimes, subregimes, time_features
            )
            
            # Store predictions and actual values
            all_predictions.extend(predictions['trend_mean'].flatten())
            all_trends.extend(trend.cpu().numpy().flatten())
            all_directions.extend(predictions['direction'].flatten())
            all_confidences.extend(predictions['confidence'].flatten())
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'trend_strength': all_trends,
        'predicted_trend': all_predictions,
        'direction': all_directions,
        'confidence': all_confidences
    })
    
    # Calculate accuracy
    results['correct'] = (np.sign(results['trend_strength']) == results['direction']).astype(int)
    accuracy = results['correct'].mean()
    
    print(f"Overall direction accuracy: {accuracy:.4f}")
    
    # Analyze predictions by direction
    up_samples = (results['trend_strength'] > 0).sum()
    down_samples = (results['trend_strength'] < 0).sum()
    
    print(f"Uptrend samples: {up_samples}, Downtrend samples: {down_samples}")
    
    up_accuracy = results.loc[results['trend_strength'] > 0, 'correct'].mean()
    down_accuracy = results.loc[results['trend_strength'] < 0, 'correct'].mean()
    
    print(f"Uptrend prediction accuracy: {up_accuracy:.4f}")
    print(f"Downtrend prediction accuracy: {down_accuracy:.4f}")
    
    # Plot results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted Trend
        plt.subplot(2, 1, 1)
        plt.plot(results['trend_strength'], 'b-', label='Actual Trend')
        plt.plot(results['predicted_trend'], 'r-', label='Predicted Trend')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Actual vs Predicted Trend Strength')
        plt.legend()
        
        # Plot 2: Direction and Confidence
        plt.subplot(2, 1, 2)
        plt.plot(results['direction'], 'g-', label='Predicted Direction')
        plt.plot(results['confidence'], 'y-', alpha=0.7, label='Prediction Confidence')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Direction Prediction and Confidence')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_results.png'))
        print(f"Plots saved to {os.path.join(output_dir, 'prediction_results.png')}")
        
        # Save results to CSV
        results.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
        print(f"Results saved to {os.path.join(output_dir, 'prediction_results.csv')}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use trained model for predictions")
    parser.add_argument("--model_dir", type=str, default="/Users/aleksandr/code/scripts/CronusV1/TTA/trained_models/models", 
                        help="Directory containing trained models")
    parser.add_argument("--data_path", type=str, default="/Users/aleksandr/code/scripts/CronusV1/TDA/regime_results.csv",
                        help="Path to data file for predictions")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to process")
    parser.add_argument("--output_dir", type=str, default="/Users/aleksandr/code/scripts/CronusV1/TTA/prediction_results",
                        help="Directory to save prediction results")
    
    args = parser.parse_args()
    
    # Load model
    model, metadata = load_best_model(args.model_dir)
    
    # Make predictions
    results = predict_trends(model, args.data_path, args.max_samples, args.output_dir) 