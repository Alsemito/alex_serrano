import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import functions

import os
import sys


def resource_path(relative_path):
    """Get the absolute path to the resource."""
    try:
        # PyInstaller stores data in _MEIPASS temporary folder
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        base_path = os.path.abspath(
            "."
        )  # Fallback to the current directory if not in PyInstaller

    return os.path.join(base_path, relative_path)


def show_frame(frame):
    """
    Clear and switch between frames

    :param frame: Frame to be displayed
    """
    global current_frame

    img_label.pack_forget()
    img_label.image = None
    result_label.config(text="")
    save_button.pack_forget()

    frame_digit_recognition.pack_forget()
    frame_object_detection.pack_forget()
    frame_keypoint_detection.pack_forget()

    frame.pack(fill=tk.BOTH, expand=True)

    # Update the current frame
    current_frame = frame
    img_label.pack()

    if current_frame != frame_digit_recognition:
        save_button.pack(side=tk.BOTTOM, pady=10)


def save_image():
    """
    Save the result image to a selected location.
    """
    global current_image

    if current_frame != frame_digit_recognition and img_label.image:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")],
        )
        if file_path:
            img = Image.fromarray(current_image)
            img.save(file_path)
            print(f"Image saved to {file_path}")


def handle_task():
    """
    Handle the selected task and display results.
    """
    global current_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not file_path:
        return
    
    file_path = resource_path(file_path)

    if current_frame == frame_digit_recognition:
        predicted_digit = functions.predict_digit(img_path=file_path)

        img = Image.open(file_path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

        # Display the predicted digit
        result_label.config(text=f"Predicted Digit: {predicted_digit}")

    elif current_frame == frame_object_detection:
        result_image = functions.detect_objects(file_path)
        image = Image.fromarray(result_image)
        image = ImageTk.PhotoImage(image)
        img_label.config(image=image)
        img_label.image = image

        current_image = result_image

    elif current_frame == frame_keypoint_detection:
        result_image = functions.face_keypoint_detection(img_path=file_path)
        image = Image.fromarray(result_image)
        img = ImageTk.PhotoImage(image)
        img_label.config(image=img)
        img_label.image = img

        current_image = result_image


# Create a root window
root = tk.Tk()
root.title("Alex Serrano's Task Selector")
root.geometry("800x600")

# Create a frame for the top bar
topbar = tk.Frame(root, bg="#2C3E50", height=50)
topbar.pack(fill=tk.X, side=tk.TOP)

# Create frames for each task
frame_digit_recognition = tk.Frame(root)
frame_object_detection = tk.Frame(root)
frame_keypoint_detection = tk.Frame(root)

# Variables to track the current frame and image
current_frame = None
current_image = None

img_label = tk.Label(root)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10, side=tk.BOTTOM)


# Create the top bar buttons
btn_digit_recognition = tk.Button(
    topbar,
    text="Digit Recognition",
    command=lambda: show_frame(frame_digit_recognition),
    relief="solid",
    fg="white",
    bg="#34495E",
    width=20,
)
btn_digit_recognition.pack(side=tk.LEFT, padx=10)

btn_object_detection = tk.Button(
    topbar,
    text="Object Detection",
    command=lambda: show_frame(frame_object_detection),
    relief="solid",
    fg="white",
    bg="#34495E",
    width=20,
)
btn_object_detection.pack(side=tk.LEFT, padx=10)

btn_keypoint_detection = tk.Button(
    topbar,
    text="Face Keypoint Detection",
    command=lambda: show_frame(frame_keypoint_detection),
    relief="solid",
    fg="white",
    bg="#34495E",
    width=20,
)
btn_keypoint_detection.pack(side=tk.LEFT, padx=10)

# Button to select an image from the device
btn_select_image = tk.Button(
    topbar,
    text="Select Image",
    command=handle_task,
    relief="solid",
    fg="white",
    bg="#34495E",
    width=20,
)
btn_select_image.pack(side=tk.RIGHT, padx=10)

# Button to save an image to the device
save_button = tk.Button(
    root,
    text="Save Image",
    command=save_image,
    relief="solid",
    fg="white",
    bg="#34495E",
    width=20,
)

label_digit_recognition = tk.Label(
    frame_digit_recognition, text="Digit Recognition Task", font=("Arial", 24)
)
label_digit_recognition.pack(pady=20)

label_keypoint_detection = tk.Label(
    frame_keypoint_detection, text="Face Keypoint Detection Task", font=("Arial", 24)
)
label_keypoint_detection.pack(pady=20)

label_object_detection = tk.Label(
    frame_object_detection, text="Object Detection Task", font=("Arial", 24)
)
label_object_detection.pack(pady=20)

# Object detection frame by default
show_frame(frame_digit_recognition)

# Start the Tkinter event loop
root.mainloop()
