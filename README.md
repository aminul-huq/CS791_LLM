# CS791_LLM

# Enviroment
Install the environment.yml to load all the necessary libraries.

# Dataset
Download the dataset from the official VQA website(https://visualqa.org/vqa_v1_download.html). Make sure to download the images, questions and annotations. We used V1 but it can be used for V2 as well. 
In the main.ipynb file adjust the paths of the downloaded files. 

# Training the Model
All the teacher and student models are initialized in the main.ipynb file. Apart from the EfficientVLM approach all other approach can be find there. If the dataset is loaded then run the program sequentially and it will train all the models.
Keep in mind that if the model saved directories are not created automatically then you would need to create them yourself. 

# Testing the Model
Use the main_test.ipynb file for testing and comparing performance. Load the dataset and location of the saved models file properly and run each cells. 

