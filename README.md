Code release for "Calibrating a Medical Imaging Classification System based on Noisy Labels"

# Basic usage
```
# Create optimizer instance

scaler_optimizer = TempScalingOnAdaECE(noisy_labels = True, epsilon = epsilon)


# Find best Temperature

opt_temperature = scaler_optimizer.find_best_T(valid_logits, torch.tensor(noisy_val_labels), optimizer='brute')


# Compute ECE on Testset with/without calibration

ece_loss = ECELoss()
ece_before = ece_loss(test_logits, torch.tensor(test_labels))
ece_after = ece_loss(test_logits/ opt_temperature, torch.tensor(test_labels))
```

# Full workflow
### Step 1: Train model
Train your classification model. Code for 3 datasets in paper are in CXR14/ham10000/path_mnist folders respecitevly.
Run 'python {cxr14/ham10000/path_mnist}/python train.py'

### Step 2: Calibrate using noisy validation set
Run different types of calibration and test over the testset.
* Original
* TS - Clean
* TS - Noise
* NTS
Run 'python noisy_calibration/calibrate.py'

