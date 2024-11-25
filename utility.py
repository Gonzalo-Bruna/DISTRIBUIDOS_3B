# My Utility : auxiliar functions
import pandas as pd
import numpy as np
import os


def read_config_sae():
    """Read SAE configuration parameters"""
    try:
        # Assuming config files are in a data directory
        config_path = os.path.join('data', 'config_sae.csv')
        config = np.loadtxt(config_path, delimiter=',')
        return {
            'hidden_nodes1': int(config[0]),  # First hidden layer nodes
            'hidden_nodes2': int(config[1]),  # Second hidden layer nodes
            # Penalty factor for pseudo-inverse
            'penalty': float(config[2]),
            'num_runs': int(config[3])        # Number of ELM runs
        }
    except Exception as e:
        raise Exception(f"Error reading config_sae.csv: {str(e)}")


def read_config_softmax():
    """Read Softmax configuration parameters"""
    try:
        config_path = os.path.join('data', 'config_softmax.csv')
        config = np.loadtxt(config_path, delimiter=',')
        return {
            'max_iter': int(config[0]),    # Maximum iterations
            'batch_size': int(config[1]),  # Mini-batch size
            'learning_rate': float(config[2])  # Learning rate
        }
    except Exception as e:
        raise Exception(f"Error reading config_softmax.csv: {str(e)}")


def load_and_preprocess_data(filename, idx_file='idx_igain.csv'):
    """Load and preprocess data using information gain indices"""
    try:
        # Adjust paths to include data directory
        data_path = os.path.join('data', filename)
        idx_path = os.path.join('data', idx_file)

        # Load data
        data = pd.read_csv(data_path, header=None)

        # Load relevant feature indices
        # -1 because indices are 1-based
        idx = pd.read_csv(idx_path, header=None).values.flatten() - 1

        # Extract features and labels
        X = data.iloc[:, :-1].values  # All columns except last
        y = data.iloc[:, -1].values   # Last column

        # Select relevant features
        X = X[:, idx]

        # Convert labels to binary format
        y_binary = label_binary(y)

        return X, y_binary
    except Exception as e:
        raise Exception(f"Error loading data {filename}: {str(e)}")


def label_binary(y):
    """Convert numeric labels to binary format
    1 -> [1,0]
    2 -> [0,1]
    """
    N = len(y)
    y_binary = np.zeros((N, 2))
    y_binary[y == 1, 0] = 1  # Normal traffic
    y_binary[y == 2, 1] = 1  # Attack traffic
    return y_binary


def sigmoid(x):
    """Sigmoid activation function with numerical stability"""
    mask = x >= 0
    out = np.zeros_like(x)
    out[mask] = 1 / (1 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    out[~mask] = exp_x / (1 + exp_x)
    return out


def calculate_pseudo_inverse(H, C):
    """Calculate pseudo-inverse using SVD"""
    try:
        # Calculate H * H^T
        HHT = H @ H.T

        # Add regularization term
        I = np.eye(HHT.shape[0])
        A = HHT + I/C

        # Compute inverse directly for numerical stability
        A_inv = np.linalg.inv(A)

        return H.T @ A_inv
    except Exception as e:
        raise Exception(f"Error in pseudo-inverse calculation: {str(e)}")


def mtx_confusion(y_true, y_pred):
    """Calculate confusion matrix and F-scores
    Returns:
    - 2x2 confusion matrix
    - F-scores for each class
    """
    # Get predicted classes
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)

    # Initialize confusion matrix
    cm = np.zeros((2, 2))
    for t, p in zip(y_true_class, y_pred_class):
        cm[t, p] += 1

    # Calculate metrics for each class
    f_scores = np.zeros(2)
    for i in range(2):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        # Calculate precision and recall
        precision = TP/(TP + FP) if (TP + FP) != 0 else 0
        recall = TP/(TP + FN) if (TP + FN) != 0 else 0

        # Calculate F-score
        f_scores[i] = 2 * (precision * recall)/(precision +
                                                recall) if (precision + recall) != 0 else 0

    return cm, f_scores


def save_outputs(costs, weights, prefix=''):
    """Save training outputs to files"""
    try:
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save costs
        cost_path = os.path.join(output_dir, f'{prefix}costo.csv')
        np.savetxt(cost_path, costs, delimiter=',', fmt='%.6f')

        # Save weights
        for i, w in enumerate(weights, 1):
            weight_path = os.path.join(output_dir, f'w{i}.npy')
            np.save(weight_path, w)

    except Exception as e:
        raise Exception(f"Error saving outputs: {str(e)}")


def load_weights():
    """Load trained weights from output directory"""
    try:
        weights = []
        for i in range(1, 4):  # Load w1, w2, w3
            weight_path = os.path.join('output', f'w{i}.npy')
            weights.append(np.load(weight_path))
        return weights
    except Exception as e:
        raise Exception(f"Error loading weights: {str(e)}")


def save_test_outputs(confusion_matrix, f_scores):
    """Save test outputs to files with"""
    try:
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save confusion matrix
        conf_path = os.path.join(output_dir, 'confusion.csv')
        np.savetxt(conf_path, confusion_matrix, delimiter=',', fmt='%d')

        # Save f-scores
        fscore_path = os.path.join(output_dir, 'fscores.csv')
        np.savetxt(fscore_path, f_scores, delimiter=',', fmt='%.4f')

    except Exception as e:
        raise Exception(f"Error saving test outputs: {str(e)}")
