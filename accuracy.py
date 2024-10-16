import numpy as np
def off_by_one_accuracy(y_true, y_pred):
    # Calculate the standard accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate the off-by-one accuracy
    off_by_one_correct = np.sum((y_true - y_pred + 1).abs() <= 1)
    off_by_one_accuracy = off_by_one_correct / len(y_true)
    
    return accuracy, off_by_one_accuracy