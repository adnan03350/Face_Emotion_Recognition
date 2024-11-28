import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog

# Paths to the models
image_model_path = 'models/image.pt'
realtime_model_path = 'models/realtime.pt'

# Load the models
try:
    image_model = YOLO(image_model_path)
    realtime_model = YOLO(realtime_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Variable to store the uploaded image path
uploaded_image_path = None
uploaded_image = None

# Function to upload and display the image
def upload_image():
    global uploaded_image_path, uploaded_image
    try:
        # Use tkinter's filedialog directly for compatibility
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not filename:  # User cancels or doesn't select a file
            status_label.configure(text="No file selected.", text_color="red")
            return

        uploaded_image_path = filename
        img = cv2.imread(filename)
        if img is None:  # Invalid image file
            status_label.configure(text="Invalid image file. Please select a valid image.", text_color="red")
            return

        # Convert the image for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the GUI
        label_image.configure(image=img_tk, text="")
        label_image.image = img_tk
        uploaded_image = img  # Store the image for detection
        status_label.configure(text="Image uploaded successfully!", text_color="green")
    except Exception as e:
        status_label.configure(text=f"Error: {str(e)}", text_color="red")

# Function to detect emotions in the uploaded image and display the result
def detect_emotion():
    global uploaded_image
    if uploaded_image is not None:
        try:
            status_label.configure(text="Starting detection...", text_color="yellow")
            # Perform prediction with the image detection model
            result = image_model.predict(source=uploaded_image, save=False)
            
            for res in result:
                img_with_detections = res.plot()  # Draw bounding boxes on the image
            
            # Convert and display the image with detections
            img_rgb_with_detections = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_with_detections)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            label_image.configure(image=img_tk, text="")
            label_image.image = img_tk  # Replace the uploaded image with the detection result
            status_label.configure(text="Detection complete!", text_color="green")
        except Exception as e:
            status_label.configure(text=f"Error during detection: {str(e)}", text_color="red")
    else:
        status_label.configure(text="No image uploaded. Please upload an image first.", text_color="red")

# Function for real-time detection
def real_time_detection():
    status_label.configure(text="Starting real-time detection...", text_color="yellow")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_label.configure(text="Error accessing webcam. Please check your camera.", text_color="red")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform prediction with the real-time detection model
            result = realtime_model.predict(source=frame, save=False)
            for res in result:
                frame = res.plot()

            cv2.imshow('Real-Time Detection', frame)

            # Press 'q' to quit the real-time detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during real-time detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        status_label.configure(text="Real-time detection stopped.", text_color="blue")

# Initialize the CustomTkinter window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("Emotion Detection System")
app.geometry("800x600")

# Title Label
title_label = ctk.CTkLabel(app, text="Emotion Detection System", font=("Arial", 24, "bold"))
title_label.pack(pady=10)

# Status Label
status_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
status_label.pack(pady=10)

# Separator
separator = ctk.CTkFrame(app, height=2, fg_color="gray")
separator.pack(fill="x", padx=20, pady=10)

# Buttons
upload_button = ctk.CTkButton(app, text="Upload Image", command=upload_image, width=200)
upload_button.pack(pady=10)

detect_button = ctk.CTkButton(app, text="Detect Emotion", command=detect_emotion, width=200)
detect_button.pack(pady=10)

real_time_button = ctk.CTkButton(app, text="Real-Time Detection", command=real_time_detection, width=200)
real_time_button.pack(pady=10)

# Image Label
label_image = ctk.CTkLabel(app, text="Upload an image to view it here.", width=500, height=300, anchor="center")
label_image.pack(pady=10)

# Run the appa
app.mainloop()
