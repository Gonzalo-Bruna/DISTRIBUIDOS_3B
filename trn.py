# Extreme Deep Learning
import numpy as np
import utility as ut
import time
import os

# Ensure the output directory exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def train_sae_layer(data, hidden_nodes, penalty, num_runs):
    """
    Train a single SAE layer using ELM with pseudo-inverse
    """
    try:
        n_samples, n_features = data.shape
        best_error = float('inf')
        best_weights = None
        best_reconstruction = None

        # Initialize weights according to the formula
        r = np.sqrt(6 / (hidden_nodes + n_features))

        for run in range(num_runs):
            # Random input weights
            w = np.random.uniform(-r, r, (n_features, hidden_nodes))

            # Calculate hidden layer output
            H = ut.sigmoid(data @ w)

            # Calculate pseudo-inverse
            H_pinv = ut.calculate_pseudo_inverse(H, penalty)

            # Calculate output weights
            w_out = H_pinv @ data

            # Calculate reconstruction error
            reconstructed = H @ w_out
            error = np.mean((data - reconstructed) ** 2)

            if error < best_error:
                best_error = error
                best_weights = w
                best_reconstruction = reconstructed

            print(f"Run {run + 1}/{num_runs}, Error: {error:.6f}")

        print(f"Best reconstruction error: {best_error:.6f}")
        return best_weights

    except Exception as e:
        raise Exception(f"Error in SAE layer training: {str(e)}")


def train_softmax(data, labels, config):
    """
    Train Softmax layer using mini-batch mAdam with numerical stability
    """
    n_features = data.shape[1]
    n_classes = labels.shape[1]

    # Initialize parameters
    w = np.random.randn(n_features, n_classes) * 0.01
    m = np.zeros_like(w)  # First moment
    v = np.zeros_like(w)  # Second moment
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    costs = []

    # Early stopping parameters
    patience = 10          # Increased patience
    min_delta = 1e-5      # Stricter convergence criterion
    best_cost = float('inf')
    patience_counter = 0
    min_epochs = 200      # Minimum epochs before allowing early stopping
    best_weights = None   # Store best weights

    n_batches = data.shape[0] // config['batch_size']

    for epoch in range(config['max_iter']):
        # Shuffle data
        idx = np.random.permutation(data.shape[0])
        data_shuffled = data[idx]
        labels_shuffled = labels[idx]

        epoch_cost = 0

        for i in range(n_batches):
            # Get mini-batch
            start_idx = i * config['batch_size']
            end_idx = start_idx + config['batch_size']
            X_batch = data_shuffled[start_idx:end_idx]
            y_batch = labels_shuffled[start_idx:end_idx]

            # Forward pass with numerical stability
            logits = X_batch @ w
            # For numerical stability
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute cost with numerical stability
            batch_cost = - \
                np.mean(np.sum(y_batch * np.log(probs + epsilon), axis=1))
            epoch_cost += batch_cost

            # Backward pass
            grad = (1/config['batch_size']) * X_batch.T @ (probs - y_batch)

            # mAdam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)

            m_hat = m / (1 - beta1**(epoch + 1))
            v_hat = v / (1 - beta2**(epoch + 1))

            w -= config['learning_rate'] * m_hat / (np.sqrt(v_hat) + epsilon)

        # Average epoch cost
        epoch_cost /= n_batches
        costs.append(epoch_cost)

        # Update best weights if current cost is better
        if epoch_cost < best_cost:
            best_cost = epoch_cost
            best_weights = w.copy()

        # Print progress
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{config['max_iter']}, Cost: {epoch_cost:.6f}")

        # Early stopping check only after minimum epochs
        if epoch >= min_epochs:
            if epoch_cost < best_cost - min_delta:
                best_cost = epoch_cost
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"Best cost achieved: {best_cost:.6f}")
                    break

    # Return the best weights instead of final weights
    return best_weights, np.array(costs)


def train_edl():
    """
    Train the complete EDL model
    """
    start_time = time.time()

    try:
        # Load configurations
        sae_config = ut.read_config_sae()
        softmax_config = ut.read_config_softmax()

        # Load and preprocess training data
        print("Loading and preprocessing data...")
        X_train, y_train = ut.load_and_preprocess_data('dtrain.csv')

        # Normalize input data and save statistics
        print("Normalizing data...")
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-10
        X_train_norm = (X_train - mean) / std

        # Save normalization parameters
        np.save('output/mean.npy', mean)
        np.save('output/std.npy', std)

        # Train first SAE layer
        print("Training first SAE layer...")
        print(f"Input shape: {X_train_norm.shape}")
        w1 = train_sae_layer(X_train_norm,
                             sae_config['hidden_nodes1'],
                             sae_config['penalty'],
                             sae_config['num_runs'])

        # Get features for second layer
        H1 = ut.sigmoid(X_train_norm @ w1)

        # Train second SAE layer
        print("Training second SAE layer...")
        print(f"Input shape: {H1.shape}")
        w2 = train_sae_layer(H1,
                             sae_config['hidden_nodes2'],
                             sae_config['penalty'],
                             sae_config['num_runs'])

        # Get features for Softmax layer
        H2 = ut.sigmoid(H1 @ w2)

        # Train Softmax layer
        print("Training Softmax layer...")
        print(f"Input shape: {H2.shape}")
        w3, costs = train_softmax(H2, y_train, softmax_config)

        # Save weights and costs
        weights = [w1, w2, w3]
        ut.save_outputs(costs, weights)

        # Check execution time
        execution_time = time.time() - start_time
        if execution_time > 120:  # 2 minutes
            print(f"Warning: Execution time ({
                  execution_time:.2f}s) exceeded 2 minutes limit")

        return weights, costs

    except Exception as e:
        print(f"Error in train_edl: {str(e)}")
        return None, None


def main():
    print("Starting EDL training...")
    weights, costs = train_edl()
    if weights is not None:
        print("Training completed successfully")
    else:
        print("Training failed")


if __name__ == '__main__':
    main()
