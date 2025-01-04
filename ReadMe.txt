Early Detection of Tomato Plant Diseases Through Image-Based Classification Methods


Overview -
This project aims to develop a machine learning-based system for the early detection and classification of tomato plant diseases using image-based methods. By leveraging deep learning models, such as Convolutional Neural Networks (CNNs), InceptionV3, and DenseNet121, the system can classify tomato leaf images into different disease categories. The goal is to help farmers and agricultural experts detect diseases at an early stage and take appropriate actions to protect their crops.

Key Features -

Disease Classification: The system classifies tomato leaf images into different disease categories, including:
- Tomato___Late_blight
- Tomato___Healthy
- Tomato___Early_blight
- Tomato___Septoria_leaf_spot
- Tomato___Tomato_Yellow_Leaf_Curl_Virus
- Tomato___Bacterial_spot
- Tomato___Target_Spot
- Tomato___Tomato_mosaic_virus
- Tomato___Leaf_Mold
- Tomato___Spider_mites Two-spotted_spider_mite

Image Preprocessing: The images are resized, normalized, and augmented to ensure the model generalizes well.
Model Variations: Several architectures are used, including basic CNN, InceptionV3, and DenseNet121 models, to determine the most effective model for tomato disease classification.
Performance Metrics: The models are evaluated based on accuracy, precision, and recall, with training and validation curves plotted for better understanding of the model's performance.
Installation
To run this project, you need to install the required libraries and dependencies. The following steps guide you through setting up the environment.

Prerequisites -
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- Scikit-learn
- Matplotlib, Seaborn, Pandas, NumPy

Workflow -
Loading the Dataset:
The images are loaded from the directories containing different tomato disease categories. These images are resized to a uniform size (224x224 pixels) for input into the deep learning models.

Model Training:
A basic Convolutional Neural Network (CNN) model is first used for training. The model architecture consists of several convolutional layers followed by fully connected layers, and the output layer has 10 units corresponding to the 10 disease classes.
Advanced models like InceptionV3 and DenseNet121 are then used with transfer learning, utilizing pre-trained weights from ImageNet to improve classification accuracy.

Model Evaluation:
The models are evaluated using accuracy, precision, and recall metrics. Training and validation curves are generated to visualize the performance of the models.

Model Results:
The final trained models are saved for inference. Visualizations of the training process (loss, accuracy, precision, recall) are also included in the outputs.

License -
This project is licensed under the MIT License
