import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Provide the model weights path separately
json_file = 'model_a.json'
weights_file = 'C:/Users/ASUS/Desktop/Emotion Detection/model.weights.h5'
model = FacialExpressionModel(json_file, weights_file)

top = tk.Tk()
top.geometry('800x600')
top.title("Emotion Detector")
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

faces = cv2.CascadeClassifier('C:/Users/ASUS/Desktop/Emotion Detection/haaracascade_frontalface_default.xml')

EMOTIONS_LIST = ["Angry","Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] 

def Detect(file_path):
    global Label_packed
    
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = faces.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in face:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        print("Predicted Emotion is:", pred)
        label1.configure(foreground="#011638", text=pred) 
    except:
        label1.configure(foreground="#011638", text="Unable to Detect")

def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_button.configure(background='#364156', foreground='White', font=('arial', 11, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfile().name
        uploaded = Image.open(file_path) 
        uploaded.thumbnail(((top.winfo_width()/2.3), top.winfo_height()/2.3)) 
        im = ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text=' ') 
        show_detect_button(file_path)
        
    except Exception as e:
        print(e)

upload = Button(top, text='Upload Image', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 25, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
heading = Label(top, text="Emotion Detector", pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156') 
heading.pack() 
top.mainloop()
