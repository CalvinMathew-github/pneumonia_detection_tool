import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input


loaded_model_densenet = load_model("model_densenet.h5")
loaded_model_efficientnet = load_model("model_efficientnet.h5")


class_labels = ['Bacterial_Pneumonia', 'Normal', 'Viral_Pneumonia', 'unknown']

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))  
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x


def pred_disease(image_path):
    x = preprocess_image('output_hsv.png')  
    
    # Get predictions from both models
    result_densenet = loaded_model_densenet.predict(x)
    result_efficientnet = loaded_model_efficientnet.predict(x)
    
    # Print the shapes of the predictions
    print("DenseNet Prediction Shape:", result_densenet.shape)
    print("EfficientNet Prediction Shape:", result_efficientnet.shape)

    # Ensure both models have the same number of output classes
    num_classes_densenet = result_densenet.shape[1]
    num_classes_efficientnet = result_efficientnet.shape[1]

    # Handling models with different output shapes
    if num_classes_densenet != num_classes_efficientnet:
        print(f"Model output shapes do not match: DenseNet ({num_classes_densenet}), EfficientNet ({num_classes_efficientnet})")
        if num_classes_densenet < num_classes_efficientnet:
            result_efficientnet = result_efficientnet[:, :num_classes_densenet]
        else:
            result_densenet = result_densenet[:, :num_classes_efficientnet]

    # Adjust class_labels to match the number of classes in the model output
    adjusted_class_labels = class_labels[:min(num_classes_densenet, num_classes_efficientnet)]
    
    # Print the probability scores for each model
    densenet_scores = {adjusted_class_labels[i]: result_densenet[0][i] * 100 for i in range(len(adjusted_class_labels))}
    efficientnet_scores = {adjusted_class_labels[i]: result_efficientnet[0][i] * 100 for i in range(len(adjusted_class_labels))}
    
    # Ensemble the predictions by averaging
    result = (result_densenet + result_efficientnet) / 2
    ensembled_scores = {adjusted_class_labels[i]: result[0][i] * 100 for i in range(len(adjusted_class_labels))}
    
    # Find the index of the highest predicted value
    final_list_result = (result * 100).astype('int')
    list_vals = list(final_list_result[0])
    result_val = max(list(final_list_result[0]))
    index_result = list_vals.index(result_val)
    
    return index_result, densenet_scores, efficientnet_scores, ensembled_scores
