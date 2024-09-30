import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array

# Load models and initialize face detector
ethnicity_model_path = r'path_of_file\ethnicity_detection_model.keras'
emotion_model_path = r'path_of_file\Emotion_model.keras'
age_model_path = r'path_of_file\age_gender_model.keras'
color_model_path = r'path_of_file\COLOR_detection_model.keras'
cascade_path = r'path_of_file\haarcascade_frontalface_default.xml'

# Age groups and ethnicity names
age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ethnicity_names = ['White', 'Black', 'Asian', 'Indian', 'Other']
emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
color_names = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

# Load pre-trained object detection model for detecting people
net = cv2.dnn.readNetFromCaffe(r"path_of_file\deploy.prototxt",
                               r"path_of_file\mobilenet_iter_73000.caffemodel")

# Function for detecting ethnicity
def detect_ethnicity(file_path, model_path, cascade_path):
    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    image = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "No face detected."

    for (x, y, w, h) in faces:
        cropped_img = img_rgb[y:y + h, x:x + w]
        resized_img = cv2.resize(cropped_img, (128, 128))
        image_array = resized_img / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        ethnicity_prediction = model.predict(image_array)
        predicted_index = np.argmax(ethnicity_prediction)
        predicted_ethnicity = ethnicity_names[predicted_index]

        # Check the confidence of the prediction
        confidence = ethnicity_prediction[0][predicted_index]
        print(f"Predicted Ethnicity: {predicted_ethnicity}, Confidence: {confidence}")

        # If the confidence is low or the prediction is frequently "White", consider the next best prediction
        if predicted_ethnicity == "White" and confidence < 0.6:
            sorted_indices = np.argsort(ethnicity_prediction[0])[::-1]  # Sort indices by confidence
            for idx in sorted_indices:
                if ethnicity_names[idx] != "White":
                    predicted_ethnicity = ethnicity_names[idx]
                    break

        return predicted_ethnicity
    
    return "Unable to detect"


# Function for detecting emotion
def detect_emotion(file_path, model_path, cascade_path):
    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return "No face detected."

    for (x, y, w, h) in faces:
        fc = gray_image[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        pred = emotion_list[np.argmax(model.predict(roi))]
        return pred

    return "Unable to detect"

# Function for detecting age
def detect_age(file_path, model_path, cascade_path):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    facec = cv2.CascadeClassifier(cascade_path)

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) == 0:
        print("Unable to detect any face.")
        return None

    for (x, y, w, h) in faces:
        fc = gray_image[y:y + h, x:x + w]
        roi = cv2.resize(fc, (128, 128))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        age_prediction = model.predict(roi)
        predicted_age = age_prediction[0][0].item()
        closest_age_group = min(age_groups, key=lambda age: abs(int(age.split('-')[0].strip('()')) - predicted_age * 100))

        print(f"Detected Age Group: {closest_age_group}")
        return closest_age_group

    print("Unable to detect any face.")
    return None

# Function for detecting cloth color
def detect_cloth_color(image_path):
    model = load_model(color_model_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return "Error: Unable to load image"

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            w, h = x1 - x, y1 - y

            if idx == 15:
                cropped_img = img_rgb[y:y + h, x:x + w]
                if cropped_img.size == 0:
                    print(f"Invalid crop at coordinates: {(x, y, w, h)}")
                    continue

                resized_img = cv2.resize(cropped_img, (128, 128))
                image_array = img_to_array(resized_img) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                prediction = model.predict(image_array)
                predicted_index = np.argmax(prediction)
                predicted_color = color_names[predicted_index]

                print(predicted_color)
                return predicted_color

    return "No person detected"

# Create GUI class
class PredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Ethnicity, Emotion, Age, and Cloth Color Detection")
        self.master.geometry("800x600")

        self.label = tk.Label(master, text="Upload an Image")
        self.label.pack()

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.result_frame = tk.Frame(master)
        self.result_frame.pack()

        self.image_label = tk.Label(self.result_frame)
        self.image_label.pack()

        self.result_text = tk.Text(self.result_frame, height=10, width=40)
        self.result_text.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.detect_attributes(file_path)

    def detect_attributes(self, image_path):
        image = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (400, 300))
        img = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        ethnicity = detect_ethnicity(image_path, ethnicity_model_path, cascade_path)
        emotion = detect_emotion(image_path, emotion_model_path, cascade_path)

        results = [
            f"Detected Ethnicity: {ethnicity}",
            f"Predicted Emotion: {emotion}"
        ]

        if ethnicity == "Indian":
            age = detect_age(image_path, age_model_path, cascade_path)
            color = detect_cloth_color(image_path)
            results.append(f"Detected Age: {age}")
            results.append(f"Detected Cloth Color: {color}")
        elif ethnicity == "White":
            age = detect_age(image_path, age_model_path, cascade_path)
            results.append(f"Detected Age: {age}")
        elif ethnicity == "Black":
            color = detect_cloth_color(image_path)
            results.append(f"Detected Cloth Color: {color}")

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, '\n'.join(results))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
