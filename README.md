
### Ethnicity, Emotion, Age, and Cloth Color Detection GUI

This Python script is a Tkinter-based graphical user interface (GUI) that allows users to upload an image and automatically detect a person's ethnicity, emotion, age, and clothing color. It leverages pre-trained deep learning models for the various tasks (ethnicity detection, emotion recognition, age estimation, and clothing color classification). 

#### Features:
1. **Ethnicity Detection**: Predicts the ethnicity of a detected face using a pre-trained ethnicity detection model. Supported ethnicities include White, Black, Asian, Indian, and Other.
2. **Emotion Detection**: Predicts the emotion of the person in the image, including emotions such as Angry, Happy, Sad, and Surprise.
3. **Age Estimation**: For certain ethnicities (Indian and White), it predicts the age group of the person.
4. **Clothing Color Detection**: For specific ethnicities (Indian and Black), it detects the dominant color of the person’s clothing using object detection and color classification models.
5. **Object Detection**: Utilizes pre-trained MobileNet models to detect people in images and focus on clothing color classification.
6. **Face Detection**: Uses OpenCV's Haar cascades to locate faces in images for ethnicity, emotion, and age prediction.

#### Usage:
- **GUI Interface**: The interface allows users to upload images through a file dialog, and the system processes the image to detect various attributes.
- **Predicted Results**: Displays ethnicity, emotion, and for applicable ethnicities, age, and clothing color in a text box below the image preview.
  
#### Workflow:
1. Upload an image via the "Upload Image" button.
2. The system detects a face, predicts the ethnicity and emotion, and for relevant ethnicities, also predicts age and clothing color.
3. The results are displayed in the text box alongside the image preview.

#### Libraries Used:
- `Tkinter`: For the GUI.
- `OpenCV`: For image processing and face detection.
- `TensorFlow / Keras`: To load the pre-trained deep learning models for ethnicity, emotion, age, and clothing color detection.
- `Pillow (PIL)`: For image handling and display in the GUI.
  
#### Models Required:
- Ethnicity detection model.
- Emotion recognition model.
- Age detection model.
- Clothing color detection model.
  
These models must be available and loaded from specified file paths in the script.

---

DATASET link: 

[UTKFace](https://susanqq.github.io/UTKFace/),
[Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) 
