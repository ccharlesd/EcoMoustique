import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Mapping des classes numériques vers des étiquettes
class_mapping = {0: 'Misc', 1: 'Tiger'}

def test_model_load():
    """Test if model can be loaded successfully"""
    try:
        model = load_model('insect_recognition_cnn_model.h5')
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Model load failed: {e}")
        return False

def test_model_prediction(model):
    """Test model prediction capabilities with visualization"""
    # Load test data
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
      
    # Random sample prediction
    sample = X_test[:10]
    true_labels = y_test[:10]
    try:
        predictions = model.predict(sample)
        predicted_labels = (predictions > 0.5).astype(int)
        
        # Assert prediction shape
        assert predictions.shape[0] == 10
        print("Prediction test passed")
        
        # Display images with true and predicted labels
        plt.figure(figsize=(10, 10))
        for i in range(10):  # Display the first 10 samples
            plt.subplot(4, 3, i + 1)
            plt.imshow(sample[i], cmap='gray')
            true_label_name = class_mapping[true_labels[i]]
            predicted_label_name = class_mapping[predicted_labels[i][0]]
            plt.title(f"True: {true_label_name}\nPred: {predicted_label_name}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        return True
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

def validate_model_performance(model, X_test, y_test):
    """
    Comprehensive model performance validation
    
    Args:
        model: Trained keras model
        X_test: Test image data
        y_test: Test labels
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Detailed metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

def run_tests():
    """Run all tests"""
    
    if test_model_load():
        model = tf.keras.models.load_model('insect_recognition_cnn_model.h5')
        test_model_prediction(model)
        # Load model and test data
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    
        validate_model_performance(model, X_test, y_test)
    else:
        print("Some tests failed. Check implementation.")

if __name__ == '__main__':
    run_tests()
