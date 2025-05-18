import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch.nn.functional as F
import math

# Import our models and data loaders
from CronusV1.TTA.enhanced_data_loader_v1 import prepare_enhanced_trading_data
from opportunity_model import TradingOpportunityModel, DirectionPredictor

# Import configurations
try:
    from model_configs import get_config, load_config, CONFIGS
    HAS_CONFIG_MODULE = True
except ImportError:
    HAS_CONFIG_MODULE = False

# Configuration
DEFAULT_CONFIG = {
    # Model parameters
    'hidden_dim': 128,        
    'regime_dim': 32,         
    'subregime_dim': 32,      
    'time_dim': 16,           
    'attention_heads': 8,     
    'dropout_rate': 0.2,
    
    # Training parameters
    'num_epochs': 20,
    'batch_size': 64,         
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'kl_weight': 0.005,       
    'contrastive_weight': 0.5, 
    'directional_weight': 0.3, 
    'focal_weight': 1.0,       # Weight for focal loss
    'grad_clip': 1.0,
    'trend_weight': 1.0,
    'early_stopping_patience': 5,
    
    # Adversarial training parameters
    'adv_start_epoch': 5,     # Start adversarial training from this epoch (later)
    'adv_epsilon': 0.002,     # Reduced perturbation size for adversarial examples
    'adv_alpha': 0.1,         # Reduced weight for adversarial loss
    'adv_scheduler': True,    # Use a scheduler for adversarial epsilon
    'adv_epsilon_min': 0.0005, # Minimum epsilon value for scheduler
    'adv_epsilon_max': 0.005,  # Maximum epsilon value for scheduler
    'adv_warmup_epochs': 3,    # Number of warmup epochs for adversarial training
    
    # Class weighting
    'use_balanced_sampling': False,  # Disable balanced sampling 
    'use_class_weights': True,      # Use class weights for loss
    'up_weight': 1.2,               # Weight for uptrend class
    'down_weight': 1.5,             # Weight for downtrend class (higher to encourage detection)
    'neutral_weight': 0.8,          # Weight for neutral class
    'balance_threshold': 0.2,       # Threshold for determining trend direction class
    
    # Loss function parameters
    'focal_gamma': 3.0,         # Increased gamma parameter for focal loss (was 2.0)
    'focal_alpha': 0.25,        # Alpha parameter for focal loss
    'auto_weight': True,        # Auto-weight classes based on distribution
    
    # Data parameters
    'window_size': 50,
    'target_horizon': 50,
    'max_samples': None,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'profit_cap': 0.05,
    
    # Signal processing parameters
    'signal_window_sizes': [8, 16, 32, 64],
    'adaptive_signal_weights': True,  # Use adaptive weights for signal processing
    
    # Direction prediction parameters
    'direction_threshold': 0.2,  # Base threshold for direction prediction
    'adaptive_threshold': True,  # Use adaptive thresholds for uptrend/downtrend
    
    # Other
    'seed': 42,
    'visualize': True,
    'disable_direction_correction': False  # Whether to disable direction correction
}

def init_weights(m):
    """Initialize weights for better training stability"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_epoch(model, dataloader, optimizer, config, epoch, device, scheduler=None):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer to use
        config: Model configuration
        epoch: Current epoch number
        device: Device to use (CPU or GPU)
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        dict: Dictionary with training metrics
    """
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # Initialize accumulators
    cum_loss = 0.0
    cum_trend_mse = 0.0
    cum_direction_acc = 0.0
    cum_kl_loss = 0.0
    cum_contrastive_loss = 0.0
    cum_directional_loss = 0.0
    cum_trend_loss = 0.0
    cum_focal_loss = 0.0
    cum_adv_loss = 0.0  # Track adversarial loss
    
    # Directional accuracy metrics
    cum_up_acc = 0.0
    cum_down_acc = 0.0
    cum_neutral_acc = 0.0
    up_count = 0
    down_count = 0
    neutral_count = 0
    
    batch_count = 0
    
    # Enable adversarial training after a certain number of epochs
    use_adversarial = epoch >= config.get('adv_start_epoch', 5)
    
    # Adversarial training parameters
    adv_scheduler = config.get('adv_scheduler', False)
    adv_epsilon_min = config.get('adv_epsilon_min', 0.0005)
    adv_epsilon_max = config.get('adv_epsilon_max', 0.005)
    adv_warmup_epochs = config.get('adv_warmup_epochs', 3)
    adv_base_epsilon = config.get('adv_epsilon', 0.002)
    adv_alpha = config.get('adv_alpha', 0.1)
    
    # Calculate epsilon based on scheduler if enabled
    if adv_scheduler and use_adversarial:
        # Scale epsilon from min to max over warmup epochs
        effective_epoch = epoch - config.get('adv_start_epoch', 5)
        if effective_epoch < adv_warmup_epochs:
            # Linear warmup
            warmup_factor = effective_epoch / adv_warmup_epochs
            adv_epsilon = adv_epsilon_min + warmup_factor * (adv_epsilon_max - adv_epsilon_min)
        else:
            # After warmup, use a cosine schedule that gradually decreases
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (effective_epoch - adv_warmup_epochs) / 
                                              (config.get('num_epochs', 20) - config.get('adv_start_epoch', 5) - adv_warmup_epochs)))
            adv_epsilon = adv_epsilon_min + cosine_factor * (adv_epsilon_max - adv_epsilon_min)
    else:
        adv_epsilon = adv_base_epsilon
    
    # Class weights for directional loss
    use_class_weights = config.get('use_class_weights', True)
    up_weight = config.get('up_weight', 1.2)
    down_weight = config.get('down_weight', 1.5)
    neutral_weight = config.get('neutral_weight', 0.8)
    
    for batch in pbar:
        batch_count += 1
        
        # Unpack batch
        features = batch['features'].to(device)
        trend = batch['trend_strength'].to(device)
        regimes = batch['regimes'].to(device) if 'regimes' in batch and batch['regimes'] is not None else None
        subregimes = batch['subregimes'].to(device) if 'subregimes' in batch and batch['subregimes'] is not None else None
        time_features = batch['time_features'].to(device) if 'time_features' in batch and batch['time_features'] is not None else None
        
        # Create direction classes for class weighting
        direction = torch.zeros_like(trend)
        balance_threshold = config.get('balance_threshold', 0.2)
        direction[trend > balance_threshold] = 1.0      # Up trend
        direction[trend < -balance_threshold] = -1.0    # Down trend
        
        # Apply class weights if enabled
        if use_class_weights:
            sample_weights = torch.ones_like(trend)
            sample_weights[direction > 0] = up_weight     # Uptrend weight
            sample_weights[direction < 0] = down_weight   # Downtrend weight
            sample_weights[direction == 0] = neutral_weight  # Neutral weight
        else:
            sample_weights = None
        
        # Standard forward pass
        optimizer.zero_grad()
        
        # Get predictions from model
        outputs = model(features, regimes, subregimes, time_features)
        trend_mean, trend_var = outputs[0], outputs[1]
        kl_loss = outputs[2]
        projected_features = outputs[3] if len(outputs) > 3 else None
        
        # Use the model's built-in loss_function with contrastive learning
        loss_dict = model.loss_function(
            trend_mean=trend_mean,
            trend_var=trend_var,
            projected_features=projected_features,
            trend_true=trend,
            kl_weight=config.get('kl_weight', 0.01),
            contrastive_weight=config.get('contrastive_weight', 0.5),
            directional_weight=config.get('directional_weight', 0.3),
            focal_weight=config.get('focal_weight', 1.0),
            sample_weights=sample_weights  # Pass sample weights to loss function
        )
        
        # Extract the total loss and component losses from the dictionary
        loss = loss_dict["loss"]
        trend_loss = loss_dict.get("trend_loss", 0.0)
        contrastive_loss = loss_dict.get("contrastive_loss", 0.0)
        directional_loss = loss_dict.get("directional_loss", 0.0)
        focal_loss = loss_dict.get("focal_loss", 0.0)
        
        adv_loss = torch.tensor(0.0, device=device)
        
        # Adversarial training for better differentiation
        if use_adversarial:
            # Generate adversarial examples
            features_adv = model.generate_adversarial_examples(
                features, regimes, subregimes, time_features, trend, epsilon=adv_epsilon
            )
            
            # Forward pass with adversarial examples
            outputs_adv = model(features_adv, regimes, subregimes, time_features)
            trend_mean_adv, trend_var_adv = outputs_adv[0], outputs_adv[1]
            projected_features_adv = outputs_adv[3] if len(outputs_adv) > 3 else None
            
            # Calculate loss on adversarial examples
            adv_loss_dict = model.loss_function(
                trend_mean=trend_mean_adv,
                trend_var=trend_var_adv,
                projected_features=projected_features_adv,
                trend_true=trend,
                kl_weight=config.get('kl_weight', 0.01),
                contrastive_weight=config.get('contrastive_weight', 0.5),
                directional_weight=config.get('directional_weight', 0.3),
                focal_weight=config.get('focal_weight', 1.0),
                sample_weights=sample_weights  # Apply same weights to adversarial examples
            )
            
            # Extract adversarial loss
            adv_loss = adv_loss_dict["loss"]
            
            # Combined loss
            loss = loss + adv_alpha * adv_loss
        
        # Backpropagate
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            mse = F.mse_loss(trend_mean, trend)
            
            # Get thresholds for direction prediction
            threshold = config.get('direction_threshold', 0.2)
            up_threshold = threshold
            down_threshold = threshold
            
            # Use direction predictor if available for adaptive thresholds
            if hasattr(model, 'direction_predictor') and model.direction_predictor is not None:
                up_threshold = model.direction_predictor.up_threshold
                down_threshold = model.direction_predictor.down_threshold
            
            # Apply thresholds to get directions
            pred_direction = torch.zeros_like(trend_mean)
            pred_direction[trend_mean > up_threshold] = 1.0      # Up trend
            pred_direction[trend_mean < -down_threshold] = -1.0  # Down trend
            
            true_direction = torch.zeros_like(trend)
            true_direction[trend > up_threshold] = 1.0      # Up trend
            true_direction[trend < -down_threshold] = -1.0  # Down trend
            
            # Calculate overall direction accuracy
            direction_acc = (pred_direction == true_direction).float().mean()
            
            # Calculate directional accuracies
            up_mask = (true_direction > 0)
            down_mask = (true_direction < 0)
            neutral_mask = (true_direction == 0)
            
            # Count samples in each direction
            up_samples = up_mask.sum().item()
            down_samples = down_mask.sum().item()
            neutral_samples = neutral_mask.sum().item()
            
            # Calculate accuracy for each direction
            if up_samples > 0:
                up_acc = (pred_direction[up_mask] == true_direction[up_mask]).float().mean().item()
                cum_up_acc += up_acc * up_samples
                up_count += up_samples
            
            if down_samples > 0:
                down_acc = (pred_direction[down_mask] == true_direction[down_mask]).float().mean().item()
                cum_down_acc += down_acc * down_samples
                down_count += down_samples
            
            if neutral_samples > 0:
                neutral_acc = (pred_direction[neutral_mask] == true_direction[neutral_mask]).float().mean().item()
                cum_neutral_acc += neutral_acc * neutral_samples
                neutral_count += neutral_samples
                
            # Update direction predictor if applicable (help it adapt during training)
            if hasattr(model, 'direction_predictor') and model.direction_predictor is not None:
                if model.direction_predictor.adaptive and epoch > 0:
                    # Only start adapting after first epoch to avoid instability
                    model.direction_predictor.update_thresholds(true_direction, trend_mean)
        
        # Track metrics
        cum_loss += loss_dict['total_loss'].item() if 'total_loss' in loss_dict else loss.item()
        cum_trend_mse += loss_dict['trend_loss'].item()
        cum_trend_loss += trend_loss.item() if isinstance(trend_loss, torch.Tensor) else trend_loss
        cum_direction_acc += direction_acc.item()
        cum_kl_loss += kl_loss.item()
        cum_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
        cum_directional_loss += directional_loss.item() if isinstance(directional_loss, torch.Tensor) else directional_loss
        cum_focal_loss += focal_loss.item() if isinstance(focal_loss, torch.Tensor) else focal_loss
        cum_adv_loss += adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
        
        # Update progress bar
        pbar.set_postfix({
            'loss': cum_loss / batch_count,
            'trend_mse': cum_trend_mse / batch_count,
            'dir_acc': cum_direction_acc / batch_count,
            'adv_eps': adv_epsilon if use_adversarial else 0
        })
    
    # Step learning rate scheduler if provided
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(cum_loss / batch_count)
        else:
            scheduler.step()
    
    # Calculate average metrics
    avg_metrics = {
        'loss': cum_loss / batch_count,
        'trend_mse': cum_trend_mse / batch_count,
        'trend_loss': cum_trend_loss / batch_count,
        'direction_acc': cum_direction_acc / batch_count,
        'kl_loss': cum_kl_loss / batch_count,
        'contrastive_loss': cum_contrastive_loss / batch_count,
        'directional_loss': cum_directional_loss / batch_count,
        'focal_loss': cum_focal_loss / batch_count,
        'adv_loss': cum_adv_loss / batch_count if use_adversarial else 0,
        'adv_epsilon': adv_epsilon if use_adversarial else 0
    }
    
    # Add directional accuracies
    if up_count > 0:
        avg_metrics['up_accuracy'] = cum_up_acc / up_count
    else:
        avg_metrics['up_accuracy'] = 0.0
        
    if down_count > 0:
        avg_metrics['down_accuracy'] = cum_down_acc / down_count
    else:
        avg_metrics['down_accuracy'] = 0.0
        
    if neutral_count > 0:
        avg_metrics['neutral_accuracy'] = cum_neutral_acc / neutral_count
    else:
        avg_metrics['neutral_accuracy'] = 0.0
        
    # Calculate class distribution
    total_samples = up_count + down_count + neutral_count
    if total_samples > 0:
        avg_metrics['up_ratio'] = up_count / total_samples
        avg_metrics['down_ratio'] = down_count / total_samples
        avg_metrics['neutral_ratio'] = neutral_count / total_samples
    
    # Calculate balanced accuracy (average of up and down accuracies)
    if up_count > 0 and down_count > 0:
        avg_metrics['balanced_accuracy'] = (avg_metrics['up_accuracy'] + avg_metrics['down_accuracy']) / 2
    elif up_count > 0:
        avg_metrics['balanced_accuracy'] = avg_metrics['up_accuracy']
    elif down_count > 0:
        avg_metrics['balanced_accuracy'] = avg_metrics['down_accuracy']
    else:
        avg_metrics['balanced_accuracy'] = 0.0
    
    return avg_metrics

def validate(model, dataloader, config, device):
    """
    Validate model on validation data
    
    Args:
        model: Model to validate
        dataloader: DataLoader with validation data
        config: Model configuration
        device: Device to use (CPU or GPU)
        
    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    cum_loss = 0.0
    cum_trend_mse = 0.0
    cum_direction_acc = 0.0
    cum_kl_loss = 0.0
    cum_contrastive_loss = 0.0
    cum_directional_loss = 0.0
    cum_trend_loss = 0.0
    cum_focal_loss = 0.0
    
    # Separate metrics for up and down directions
    cum_up_acc = 0.0
    cum_down_acc = 0.0
    cum_neutral_acc = 0.0
    up_count = 0
    down_count = 0
    neutral_count = 0
    
    batch_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_count += 1
            
            # Unpack batch
            features = batch['features'].to(device)
            trend = batch['trend_strength'].to(device)
            regimes = batch['regimes'].to(device) if 'regimes' in batch and batch['regimes'] is not None else None
            subregimes = batch['subregimes'].to(device) if 'subregimes' in batch and batch['subregimes'] is not None else None
            time_features = batch['time_features'].to(device) if 'time_features' in batch and batch['time_features'] is not None else None
            
            # Get predictions from model
            outputs = model(features, regimes, subregimes, time_features)
            trend_mean, trend_var = outputs[0], outputs[1]
            kl_loss = outputs[2]
            projected_features = outputs[3] if len(outputs) > 3 else None
            
            # Use the model's built-in loss_function with contrastive learning
            loss_dict = model.loss_function(
                trend_mean=trend_mean,
                trend_var=trend_var,
                projected_features=projected_features,
                trend_true=trend,
                kl_weight=config.get('kl_weight', 0.01),
                contrastive_weight=config.get('contrastive_weight', 0.5),
                directional_weight=config.get('directional_weight', 0.3),
                focal_weight=config.get('focal_weight', 1.0)
            )
            
            # Extract the total loss and component losses from the dictionary
            loss = loss_dict["loss"]
            trend_loss = loss_dict.get("trend_loss", 0.0)
            contrastive_loss = loss_dict.get("contrastive_loss", 0.0)
            directional_loss = loss_dict.get("directional_loss", 0.0)
            focal_loss = loss_dict.get("focal_loss", 0.0)
            
            # Calculate metrics
            mse = F.mse_loss(trend_mean, trend)
            
            # Get predicted and true directions
            threshold = config.get('direction_threshold', 0.2)
            up_threshold = threshold
            down_threshold = threshold
            
            # Use direction predictor if available for adaptive thresholds
            if hasattr(model, 'direction_predictor') and model.direction_predictor is not None:
                up_threshold = model.direction_predictor.up_threshold
                down_threshold = model.direction_predictor.down_threshold
            
            # Predict directions using thresholds
            pred_direction = torch.zeros_like(trend_mean)
            pred_direction[trend_mean > up_threshold] = 1.0      # Up trend
            pred_direction[trend_mean < -down_threshold] = -1.0  # Down trend
            
            true_direction = torch.zeros_like(trend)
            true_direction[trend > up_threshold] = 1.0      # Up trend
            true_direction[trend < -down_threshold] = -1.0  # Down trend
            
            # Calculate overall direction accuracy
            direction_acc = (pred_direction == true_direction).float().mean()
            
            # Calculate directional accuracies
            up_mask = (true_direction > 0)
            down_mask = (true_direction < 0)
            neutral_mask = (true_direction == 0)
            
            # Count samples in each direction
            up_samples = up_mask.sum().item()
            down_samples = down_mask.sum().item()
            neutral_samples = neutral_mask.sum().item()
            
            # Calculate accuracy for each direction
            if up_samples > 0:
                up_acc = (pred_direction[up_mask] == true_direction[up_mask]).float().mean().item()
                cum_up_acc += up_acc * up_samples
                up_count += up_samples
            
            if down_samples > 0:
                down_acc = (pred_direction[down_mask] == true_direction[down_mask]).float().mean().item()
                cum_down_acc += down_acc * down_samples
                down_count += down_samples
            
            if neutral_samples > 0:
                neutral_acc = (pred_direction[neutral_mask] == true_direction[neutral_mask]).float().mean().item()
                cum_neutral_acc += neutral_acc * neutral_samples
                neutral_count += neutral_samples
            
            # Track metrics
            cum_loss += loss_dict['total_loss'].item() if 'total_loss' in loss_dict else loss.item()
            cum_trend_mse += loss_dict['trend_loss'].item()
            cum_trend_loss += trend_loss.item() if isinstance(trend_loss, torch.Tensor) else trend_loss
            cum_direction_acc += direction_acc.item()
            cum_kl_loss += kl_loss.item()
            cum_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
            cum_directional_loss += directional_loss.item() if isinstance(directional_loss, torch.Tensor) else directional_loss
            cum_focal_loss += focal_loss.item() if isinstance(focal_loss, torch.Tensor) else focal_loss
    
    # Calculate average metrics
    avg_metrics = {
        'loss': cum_loss / batch_count,
        'trend_mse': cum_trend_mse / batch_count,
        'trend_loss': cum_trend_loss / batch_count,
        'direction_acc': cum_direction_acc / batch_count,
        'kl_loss': cum_kl_loss / batch_count,
        'contrastive_loss': cum_contrastive_loss / batch_count,
        'directional_loss': cum_directional_loss / batch_count,
        'focal_loss': cum_focal_loss / batch_count
    }
    
    # Add directional accuracies
    if up_count > 0:
        avg_metrics['up_accuracy'] = cum_up_acc / up_count
    else:
        avg_metrics['up_accuracy'] = 0.0
        
    if down_count > 0:
        avg_metrics['down_accuracy'] = cum_down_acc / down_count
    else:
        avg_metrics['down_accuracy'] = 0.0
        
    if neutral_count > 0:
        avg_metrics['neutral_accuracy'] = cum_neutral_acc / neutral_count
    else:
        avg_metrics['neutral_accuracy'] = 0.0
        
    # Calculate class counts
    avg_metrics['up_count'] = up_count
    avg_metrics['down_count'] = down_count
    avg_metrics['neutral_count'] = neutral_count
    
    # Calculate balanced accuracy (average of up and down accuracies)
    if up_count > 0 and down_count > 0:
        avg_metrics['balanced_accuracy'] = (avg_metrics['up_accuracy'] + avg_metrics['down_accuracy']) / 2
    elif up_count > 0:
        avg_metrics['balanced_accuracy'] = avg_metrics['up_accuracy']
    elif down_count > 0:
        avg_metrics['balanced_accuracy'] = avg_metrics['down_accuracy']
    else:
        avg_metrics['balanced_accuracy'] = 0.0
    
    return avg_metrics

def test_model(model, test_loader, save_dir, device):
    """Test the model and print results focusing on direction prediction accuracy"""
    model.eval()
    
    # Initialize metrics
    test_loss = 0
    test_trend_mse = 0
    test_dir_acc = 0
    num_samples = 0
    
    # Storage for trends and predictions
    all_trends = []
    all_predictions = []
    all_variances = []
    all_directions = []
    true_directions = []
    
    # Directional accuracies
    up_correct = 0
    down_correct = 0
    neutral_correct = 0
    up_total = 0
    down_total = 0
    neutral_total = 0
    
    print("\nTesting model...")
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch
            features = batch['features'].to(device)
            trend = batch['trend_strength'].to(device)
            regimes = batch['regimes'].to(device) if 'regimes' in batch else None
            subregimes = batch['subregimes'].to(device) if 'subregimes' in batch else None
            time_features = batch['time_features'].to(device) if 'time_features' in batch else None
            
            # Get predictions with calibrated direction predictor
            predictions = model.predict(
                features, regimes, subregimes, time_features, 
                calibrate=True,  # Use adaptive thresholds 
                true_direction=torch.sign(trend)  # Provide true directions for calibration
            )
            
            # Get thresholds (for logging)
            up_threshold = predictions.get('up_threshold', 0.2)
            down_threshold = predictions.get('down_threshold', 0.2)
            
            # Calculate metrics
            trend_mean = torch.tensor(predictions['trend_mean']).to(device)
            trend_var = torch.tensor(predictions['trend_std']).to(device) ** 2
            trend_mse = F.mse_loss(trend_mean, trend)
            
            # True directions using thresholds
            true_direction = torch.zeros_like(trend)
            true_direction[trend > up_threshold] = 1.0
            true_direction[trend < -down_threshold] = -1.0
            
            # Use predicted directions from the model
            pred_direction = torch.tensor(predictions['direction']).to(device)
            
            # Calculate direction accuracy
            dir_accuracy = (pred_direction == true_direction).float().mean()
            
            # Directional accuracies
            up_mask = (true_direction > 0)
            down_mask = (true_direction < 0)
            neutral_mask = (true_direction == 0)
            
            # Update counts
            up_total += up_mask.sum().item()
            down_total += down_mask.sum().item()
            neutral_total += neutral_mask.sum().item()
            
            # Update correct predictions
            if up_mask.sum() > 0:
                up_correct += ((pred_direction == true_direction) & up_mask).sum().item()
            if down_mask.sum() > 0:
                down_correct += ((pred_direction == true_direction) & down_mask).sum().item()
            if neutral_mask.sum() > 0:
                neutral_correct += ((pred_direction == true_direction) & neutral_mask).sum().item()
            
            # Store metrics
            batch_size = features.size(0)
            test_trend_mse += trend_mse.item() * batch_size
            test_dir_acc += dir_accuracy.item() * batch_size
            num_samples += batch_size
            
            # Store for analysis
            all_trends.append(trend.cpu().numpy())
            all_predictions.append(predictions['trend_mean'])
            all_variances.append(predictions['trend_std']**2)
            all_directions.append(predictions['direction'])
            true_directions.append(true_direction.cpu().numpy())
    
    # Calculate metrics
    test_trend_mse /= num_samples
    test_dir_acc /= num_samples
    test_loss = test_trend_mse  # Use trend MSE as the main test loss
    
    # Concatenate all arrays
    all_trends = np.concatenate(all_trends)
    all_predictions = np.concatenate(all_predictions)
    all_variances = np.concatenate(all_variances)
    all_directions = np.concatenate(all_directions)
    true_directions = np.concatenate(true_directions)
    
    # Calculate direction prediction metrics
    positive_rate = np.mean(all_directions > 0)
    negative_rate = np.mean(all_directions < 0)
    neutral_rate = np.mean(all_directions == 0)
    
    # Calculate uptrend and downtrend accuracy
    up_accuracy = up_correct / max(up_total, 1)
    down_accuracy = down_correct / max(down_total, 1)
    neutral_accuracy = neutral_correct / max(neutral_total, 1)
    
    # Calculate balanced accuracy
    balanced_accuracy = (up_accuracy + down_accuracy) / 2
    
    # Print results
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.6f}")
    print(f"Trend MSE: {test_trend_mse:.6f}")
    print(f"Direction Accuracy: {test_dir_acc*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy*100:.2f}%")
    print(f"\nDirection Prediction Metrics:")
    print(f"Uptrend Predictions: {positive_rate*100:.2f}% (Threshold: {up_threshold:.4f})")
    print(f"Downtrend Predictions: {negative_rate*100:.2f}% (Threshold: {down_threshold:.4f})")
    print(f"Neutral Predictions: {neutral_rate*100:.2f}%")
    print(f"Uptrend Accuracy: {up_accuracy*100:.2f}% ({up_correct}/{up_total})")
    print(f"Downtrend Accuracy: {down_accuracy*100:.2f}% ({down_correct}/{down_total})")
    print(f"Neutral Accuracy: {neutral_accuracy*100:.2f}% ({neutral_correct}/{neutral_total})")
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'results_summary_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Loss: {test_loss:.6f}\n")
        f.write(f"Trend MSE: {test_trend_mse:.6f}\n")
        f.write(f"Direction Accuracy: {test_dir_acc*100:.2f}%\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy*100:.2f}%\n")
        f.write(f"\nDirection Prediction Metrics:\n")
        f.write(f"Uptrend Predictions: {positive_rate*100:.2f}% (Threshold: {up_threshold:.4f})\n")
        f.write(f"Downtrend Predictions: {negative_rate*100:.2f}% (Threshold: {down_threshold:.4f})\n")
        f.write(f"Neutral Predictions: {neutral_rate*100:.2f}%\n")
        f.write(f"Uptrend Accuracy: {up_accuracy*100:.2f}% ({up_correct}/{up_total})\n")
        f.write(f"Downtrend Accuracy: {down_accuracy*100:.2f}% ({down_correct}/{down_total})\n")
        f.write(f"Neutral Accuracy: {neutral_accuracy*100:.2f}% ({neutral_correct}/{neutral_total})\n")
    
    return {
        'loss': test_loss,
        'trend_mse': test_trend_mse,
        'dir_acc': test_dir_acc,
        'balanced_acc': balanced_accuracy,
        'positive_rate': positive_rate,
        'negative_rate': negative_rate,
        'neutral_rate': neutral_rate,
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'neutral_accuracy': neutral_accuracy,
        'up_correct': up_correct,
        'down_correct': down_correct,
        'neutral_correct': neutral_correct,
        'up_total': up_total,
        'down_total': down_total,
        'neutral_total': neutral_total,
        'trend_preds': all_predictions,
        'trend_targets': all_trends,
        'up_threshold': up_threshold,
        'down_threshold': down_threshold
    }

def loss_function(trend_mean, trend_var, trend_true, kl_loss, projected_features, config):
    """
    Custom loss function combining trend strength loss, directional loss, and KL divergence
    
    Args:
        trend_mean (torch.Tensor): Predicted mean trend strength
        trend_var (torch.Tensor): Predicted variance of trend strength
        trend_true (torch.Tensor): True trend strength values
        kl_loss (torch.Tensor): KL divergence loss from Bayesian model
        projected_features (torch.Tensor): Projected feature representations
        config (dict): Configuration dictionary with loss weights
    
    Returns:
        torch.Tensor: Combined loss value
    """
    # Get relevant config parameters with defaults
    kl_weight = config.get('kl_weight', 0.01)
    contrastive_weight = config.get('contrastive_weight', 0.5)
    directional_weight = config.get('directional_weight', 0.3)
    
    # Use the model's built-in loss_function
    # This will calculate the Gaussian NLL, contrastive loss, and directional contrastive loss
    # Need to call this from the model, so we'll modify the train_epoch and validate functions
    
    # For compatibility with existing code, we just calculate a trend loss here
    # The actual contrastive implementation is handled in train_epoch and validate
    
    # Calculate negative log likelihood (NLL) for Gaussian distribution
    # For a Gaussian, NLL = 0.5 * (log(2πσ²) + (y - μ)²/σ²)
    epsilon = 1e-6  # To avoid numerical instability
    nll = 0.5 * (torch.log(2 * math.pi * trend_var + epsilon) + 
                 (trend_true - trend_mean)**2 / (trend_var + epsilon))
    trend_loss = nll.mean()
    
    # Just return the Gaussian NLL - model's built-in loss_function will be used in train_epoch
    return trend_loss

def train_model(config, data_path, save_dir):
    """Main training function"""
    # Create save directories
    model_dir = os.path.join(save_dir, 'models')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Analyze regimes in the data first
    print("\nAnalyzing regimes in the data...")
    df = pd.read_csv(data_path)
    unique_regimes = sorted(df['regime'].unique().tolist())
    unique_subregimes = sorted(df['sub_regime'].unique().tolist())
    
    max_regime_id = max(unique_regimes)
    max_subregime_id = max(unique_subregimes)
    
    print(f"Found {len(unique_regimes)} unique regimes: {unique_regimes}")
    print(f"Found {len(unique_subregimes)} unique subregimes: {unique_subregimes}")
    print(f"Max regime ID: {max_regime_id}, Max subregime ID: {max_subregime_id}")
    
    # Create a mapping from regime/subregime values to embedding indices
    # We'll use a continuous numbering from 0 to N-1 for embedding indices
    regime_mapping = {regime: idx for idx, regime in enumerate(unique_regimes)}
    subregime_mapping = {subregime: idx for idx, subregime in enumerate(unique_subregimes)}
    
    # Create inverse mapping for model initialization
    inverse_regime_mapping = {idx: regime for regime, idx in regime_mapping.items()}
    inverse_subregime_mapping = {idx: subregime for subregime, idx in subregime_mapping.items()}
    
    print(f"Regime mapping: {regime_mapping}")
    print(f"Subregime mapping: {subregime_mapping}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, data_metadata = prepare_enhanced_trading_data(
        csv_path=data_path,
        window_size=config['window_size'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        target_horizon=config['target_horizon'],
        profit_cap=config['profit_cap'],
        max_samples=config['max_samples'],
        random_seed=config['seed'],
        visualize=config['visualize'],
        signal_window_sizes=config['signal_window_sizes'],
        results_dir=results_dir,
        regime_mapping=regime_mapping,
        subregime_mapping=subregime_mapping,
        disable_direction_correction=config['disable_direction_correction'],
        use_balanced_sampling=config['use_balanced_sampling'],
        balance_threshold=config['balance_threshold']
    )
    
    print(f"Data loaded: {data_metadata['train_size']} train, "
          f"{data_metadata['val_size']} validation, "
          f"{data_metadata['test_size']} test samples")
    
    # Create model
    print("\nCreating model...")
    input_dim = data_metadata['feature_dim']
    print(f"Input dimension: {input_dim}")
    
    model = TradingOpportunityModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        regime_dim=config['regime_dim'],
        subregime_dim=config['subregime_dim'],
        time_dim=config['time_dim'],
        n_heads=config['attention_heads'],
        n_regimes=len(unique_regimes),
        n_subregimes=len(unique_subregimes),
        dropout=config['dropout_rate'],
        device=device
    ).to(device)
    
    # Initialize weights
    model.apply(init_weights)
    
    # Set the regime and subregime mapping attributes
    model.regime_mapping = regime_mapping
    model.subregime_mapping = subregime_mapping
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create learning rate scheduler
    if config.get('use_lr_schedule', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.get('lr_schedule_factor', 0.5), 
            patience=config.get('lr_schedule_patience', 3), 
            verbose=True,
            min_lr=1e-6
        )
        print(f"Using learning rate scheduler with patience {config.get('lr_schedule_patience', 3)} "
              f"and factor {config.get('lr_schedule_factor', 0.5)}")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    direction_accuracies = []
    contrastive_losses = []
    directional_losses = []
    balanced_accuracies = []
    up_accuracies = []
    down_accuracies = []
    best_val_loss = float('inf')
    best_balanced_acc = 0.0
    patience_counter = 0
    best_model_path = None  # Track the path of the best model
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, epoch, device, scheduler)
        train_losses.append(train_metrics['loss'])
        contrastive_losses.append(train_metrics['contrastive_loss'])
        directional_losses.append(train_metrics['directional_loss'])
        
        print(f"Train Loss: {train_metrics['loss']:.6f}, "
              f"Trend MSE: {train_metrics['trend_mse']:.6f}, "
              f"Dir Acc: {train_metrics['direction_acc']:.6f}, "
              f"Balanced Acc: {train_metrics.get('balanced_accuracy', 0.0):.6f}")
        print(f"  Up Acc: {train_metrics.get('up_accuracy', 0.0):.6f}, "
              f"Down Acc: {train_metrics.get('down_accuracy', 0.0):.6f}")
        
        # Validate
        val_metrics = validate(model, val_loader, config, device)
        val_losses.append(val_metrics['loss'])
        direction_accuracies.append(val_metrics['direction_acc'])
        balanced_accuracies.append(val_metrics.get('balanced_accuracy', 0.0))
        up_accuracies.append(val_metrics.get('up_accuracy', 0.0))
        down_accuracies.append(val_metrics.get('down_accuracy', 0.0))
        
        print(f"Val Loss: {val_metrics['loss']:.6f}, "
              f"Trend MSE: {val_metrics['trend_mse']:.6f}, "
              f"Direction Acc: {val_metrics['direction_acc']:.2%}, "
              f"Balanced Acc: {val_metrics.get('balanced_accuracy', 0.0):.2%}")
        print(f"  Up Acc: {val_metrics.get('up_accuracy', 0.0):.2%}, "
              f"Down Acc: {val_metrics.get('down_accuracy', 0.0):.2%}")
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Use balanced accuracy for learning rate scheduling (negative because scheduler expects loss)
            scheduler.step(-val_metrics.get('balanced_accuracy', 0.0))
        else:
            scheduler.step()
        
        # Check for model saving based on balanced accuracy
        current_balanced_acc = val_metrics.get('balanced_accuracy', 0.0)
        improved_balanced_acc = current_balanced_acc > best_balanced_acc
        improved_loss = val_metrics['loss'] < best_val_loss
        
        # Save model if it has better balanced accuracy OR significantly better loss with similar balanced acc
        if improved_balanced_acc or (improved_loss and current_balanced_acc >= 0.95 * best_balanced_acc):
            if improved_balanced_acc:
                best_balanced_acc = current_balanced_acc
            if improved_loss:
                best_val_loss = val_metrics['loss']
                
            patience_counter = 0
            
            # Save best model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}_{timestamp}.pt')
            best_model_path = model_path  # Save the path for later use
            
            model.save_model(model_path, {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_trend_mse': val_metrics['trend_mse'],
                'direction_accuracy': val_metrics['direction_acc'],
                'balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0),
                'up_accuracy': val_metrics.get('up_accuracy', 0.0),
                'down_accuracy': val_metrics.get('down_accuracy', 0.0),
                'config': config,
                'optimizer_state': optimizer.state_dict()
            })
            
            print(f"Model saved to {model_path} (balanced acc: {current_balanced_acc:.4f}, loss: {val_metrics['loss']:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curves
    plt.figure(figsize=(15, 12))
    
    # Plot loss curves
    plt.subplot(3, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot overall direction accuracy and balanced accuracy
    plt.subplot(3, 2, 2)
    plt.plot(range(1, len(direction_accuracies) + 1), direction_accuracies, 'g-o', label='Direction Accuracy')
    plt.plot(range(1, len(balanced_accuracies) + 1), balanced_accuracies, 'm-o', label='Balanced Accuracy')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Direction Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot directional accuracies (up/down)
    plt.subplot(3, 2, 3)
    plt.plot(range(1, len(up_accuracies) + 1), up_accuracies, 'g-o', label='Uptrend Accuracy')
    plt.plot(range(1, len(down_accuracies) + 1), down_accuracies, 'r-o', label='Downtrend Accuracy')
    plt.axhline(y=0.5, color='k', linestyle='--', label='Random Guess')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Directional Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot contrastive loss
    plt.subplot(3, 2, 4)
    plt.plot(range(1, len(contrastive_losses) + 1), contrastive_losses, 'b-o', label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot directional loss
    plt.subplot(3, 2, 5)
    plt.plot(range(1, len(directional_losses) + 1), directional_losses, 'm-o', label='Directional Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Directional Contrastive Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(results_dir, f'training_curves_{timestamp}.png'))
    plt.close()
    
    # Test best model
    print("\nLoading best model for testing...")
    if best_model_path is None or not os.path.exists(best_model_path):
        print(f"Warning: Best model path {best_model_path} not found. Using latest model.")
        # Find the latest model file as fallback
        model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('model_epoch_')])
        if model_files:
            best_model_path = os.path.join(model_dir, model_files[-1])
            print(f"Using {best_model_path} instead.")
        else:
            print("No model files found. Skipping testing.")
            return model, None
    
    best_model, metadata = TradingOpportunityModel.load_model(best_model_path, device)
    
    print("Running test evaluation...")
    test_metrics = test_model(best_model, test_loader, results_dir, device)
    
    # Save results summary
    results_summary = {
        'config': config,
        'test_metrics': test_metrics,
        'data_metadata': data_metadata,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_path = os.path.join(results_dir, f'results_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Results Summary\n")
        f.write("=======================\n\n")
        
        # Write configuration
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Write test metrics
        f.write("Test Metrics:\n")
        for key, value in test_metrics.items():
            if key not in ['trend_preds', 'trend_targets']:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Calculate and display directional accuracy for the test set
        trend_preds = test_metrics['trend_preds']
        trend_targets = test_metrics['trend_targets']
        
        pred_direction = np.sign(trend_preds)
        true_direction = np.sign(trend_targets)
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        f.write(f"  Direction Accuracy: {direction_accuracy:.2%}\n\n")
        
        # Write direction prediction metrics
        f.write("  Direction Prediction Metrics:\n")
        f.write(f"    Uptrend Predictions: {test_metrics['positive_rate']:.2%}\n")
        f.write(f"    Downtrend Predictions: {test_metrics['negative_rate']:.2%}\n")
        f.write(f"    Neutral Predictions: {test_metrics['neutral_rate']:.2%}\n")
        f.write(f"    Uptrend Accuracy: {test_metrics['up_accuracy']:.2%}\n")
        f.write(f"    Downtrend Accuracy: {test_metrics['down_accuracy']:.2%}\n")
        f.write(f"    Neutral Accuracy: {test_metrics['neutral_accuracy']:.2%}\n")
        
        # Write data information
        f.write("Data Information:\n")
        for key, value in data_metadata.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"    {sub_key}: {sub_value}\n")
            else:
                f.write(f"  {key}: {value}\n")
    
    print(f"Results summary saved to {summary_path}")
    return best_model, test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trend Prediction Model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--window_size', type=int, default=50, help='Window size for sequence data')
    parser.add_argument('--target_horizon', type=int, default=50, help='Target prediction horizon')
    parser.add_argument('--direction_threshold', type=float, default=0.2, help='Threshold for direction prediction')
    parser.add_argument('--disable_direction_correction', action='store_true', 
                       help='Disable automatic correction of trend direction based on correlation')
    args = parser.parse_args()
    
    # Set directories
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Get default configuration and update with arguments
    if HAS_CONFIG_MODULE:
        config = get_config('default')
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Update config with command line arguments
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'window_size': args.window_size,
        'target_horizon': args.target_horizon,
        'max_samples': args.max_samples,
        'direction_threshold': args.direction_threshold,
        'disable_direction_correction': args.disable_direction_correction
    })
    
    # Train the model
    best_model, test_metrics = train_model(config, args.data_path, args.save_dir) 