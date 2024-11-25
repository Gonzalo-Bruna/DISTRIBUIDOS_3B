# Testing : EDL
import numpy as np
import utility as ut
import time


def forward_edl():
    """
    Forward pass through the trained EDL network
    Returns predictions for test data
    """
    # Start timing
    start_time = time.time()

    try:
        # Load test data
        print("Loading test data...")
        X_test, y_test = ut.load_and_preprocess_data('dtest.csv')

        # Load trained weights and normalization parameters
        print("Loading weights and normalization parameters...")
        weights = ut.load_weights()
        w1, w2, w3 = weights

        # Load normalization parameters
        try:
            mean = np.load('output/mean.npy')
            std = np.load('output/std.npy')
        except Exception as e:
            raise Exception(
                f"Error loading normalization parameters: {str(e)}")

        # Normalize test data using training statistics
        print("Normalizing test data...")
        X_test_norm = (X_test - mean) / std

        print("Performing forward pass...")
        # First SAE layer forward pass
        H1 = ut.sigmoid(X_test_norm @ w1)

        # Second SAE layer forward pass
        H2 = ut.sigmoid(H1 @ w2)

        # Softmax layer forward pass with numerical stability
        logits = H2 @ w3
        # Subtract max for numerical stability
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Calculate confusion matrix and F-scores
        print("Calculating metrics...")
        conf_matrix, f_scores = ut.mtx_confusion(y_test, predictions)

        # Calculate and print additional metrics
        print("\nPerformance Metrics:")
        # Calculate accuracy
        accuracy = (conf_matrix[0, 0] +
                    conf_matrix[1, 1]) / np.sum(conf_matrix)
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Calculate metrics for each class
        for i, class_name in enumerate(['Normal', 'Attack']):
            precision = conf_matrix[i, i] / np.sum(conf_matrix[:, i])
            recall = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
            f1 = f_scores[i]
            print(f"\n{class_name} Class Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

        # Save outputs
        print("\nSaving results...")
        ut.save_test_outputs(conf_matrix, f_scores)

        # Check execution time
        execution_time = time.time() - start_time
        if execution_time > 30:
            print(f"Warning: Execution time ({
                  execution_time:.2f}s) exceeded 30 seconds limit")

        return predictions, conf_matrix, f_scores

    except Exception as e:
        print(f"Error in forward_edl: {str(e)}")
        return None, None, None


def main():
    print("Starting EDL testing...")
    predictions, conf_matrix, f_scores = forward_edl()
    if predictions is not None:
        print("\nTesting completed successfully")

        # Print confusion matrix in a more readable format
        print("\nConfusion Matrix:")
        print("                 Predicted Normal  Predicted Attack")
        print(f"Actual Normal    {conf_matrix[0, 0]:^15.0f} {
              conf_matrix[0, 1]:^16.0f}")
        print(f"Actual Attack    {conf_matrix[1, 0]:^15.0f} {
              conf_matrix[1, 1]:^16.0f}")
    else:
        print("Testing failed")


if __name__ == '__main__':
    main()
