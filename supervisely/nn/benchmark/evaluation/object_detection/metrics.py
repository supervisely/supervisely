import numpy as np


def expected_calibration_error(y_true, pred_scores, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).

    Parameters:
    y_true (array-like): True binary labels.
    pred_scores (array-like): Predicted probabilities for the positive class.
    n_bins (int): Number of bins to use for the calculation. Default is 10.

    Returns:
    float: Expected Calibration Error (ECE).
    """
    
    # Ensure the inputs are numpy arrays
    y_true = np.asarray(y_true)
    pred_scores = np.asarray(pred_scores)
    
    # Initialize variables
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    
    # Compute the ECE
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(pred_scores > bin_lower, pred_scores <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(pred_scores[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def maximum_calibration_error(y_true, pred_scores, n_bins=10):
    """
    Compute the Maximum Calibration Error (MCE).

    Parameters:
    y_true (array-like): True binary labels.
    pred_scores (array-like): Predicted probabilities for the positive class.
    n_bins (int): Number of bins to use for the calculation. Default is 10.

    Returns:
    float: Maximum Calibration Error (MCE).
    """
    
    # Ensure the inputs are numpy arrays
    y_true = np.asarray(y_true)
    pred_scores = np.asarray(pred_scores)
    
    # Initialize variables
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    max_calibration_error = 0.0
    
    # Compute the MCE
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(pred_scores > bin_lower, pred_scores <= bin_upper)
        
        if np.any(in_bin):
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(pred_scores[in_bin])
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, bin_error)
            
    return max_calibration_error
