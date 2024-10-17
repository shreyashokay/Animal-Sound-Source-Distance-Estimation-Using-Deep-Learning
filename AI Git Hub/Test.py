import numpy as np
import tensorflow as tf
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox

# Set parameters
duration = 5  # Recording duration in seconds
sample_rate = 16000  # Sample rate expected by the model
expected_input_length = 44032  # The expected length of input by the model

# Paths to the model and label files
model_path = r"C:\Users\shrey\Desktop\AI project\new2\soundclassifier_with_metadata.tflite"
label_file_path = r"C:\Users\shrey\Desktop\AI project\new file\labels.txt"

# Load the TensorFlow Lite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels
with open(label_file_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Function to adjust the recorded audio length
def adjust_audio_length(audio_data, expected_length):
    current_length = len(audio_data)
    if current_length > expected_length:
        # Trim the audio if it's too long
        audio_data = audio_data[:expected_length]
    elif current_length < expected_length:
        # Pad the audio with zeros if it's too short
        padding = expected_length - current_length
        audio_data = np.pad(audio_data, (0, padding), 'constant')
    return audio_data

# Function to preprocess the recorded audio
def preprocess_audio(audio_data):
    # Adjust the shape and data type according to the model's input
    audio_data = adjust_audio_length(audio_data, expected_input_length)
    audio_data = np.expand_dims(audio_data, axis=0).astype(np.float32)
    return audio_data

# Function to run inference on the audio data
def predict(audio_data):
    # Preprocess the audio
    audio_input = preprocess_audio(audio_data)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Find the predicted label
    predicted_label_idx = np.argmax(output_data)
    predicted_label = labels[predicted_label_idx]
    
    return predicted_label

# Record audio from the microphone
def record_audio(duration, sample_rate):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    return audio_data.flatten()

# Function to handle the recording and prediction process
def start_recording():
    try:
        # Disable the start button during the process
        start_button.config(state=tk.DISABLED)
        prediction_label.config(text="")
        countdown_label.config(text=f"Recording in progress...")

        # Countdown before and during the recording
        countdown(duration)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        start_button.config(state=tk.NORMAL)

# Function to show the countdown while recording
def countdown(remaining_time):
    if remaining_time > 0:
        countdown_label.config(text=f"Recording... {remaining_time} seconds left")
        root.after(1000, countdown, remaining_time - 1)
    else:
        # When countdown ends, record and process the audio
        audio_data = record_audio(duration, sample_rate)
        process_prediction(audio_data)

# Function to process and display the prediction
def process_prediction(audio_data):
    countdown_label.config(text="Processing...")
    root.after(500, show_prediction, audio_data)  # Wait briefly to simulate processing time

# Function to handle displaying the prediction results
def show_prediction(audio_data):
    # Run prediction
    prediction = predict(audio_data)

    # Update the GUI with the prediction
    prediction_label.config(text=f"Prediction: {prediction}")

    countdown_label.config(text="Recording complete.")
    start_button.config(state=tk.NORMAL)

# Create the GUI
root = tk.Tk()
root.title("Real-time Audio Source Distance Estimation")
root.geometry("400x300")  # Set window size

# Styling
bg_color = "#f0f8ff"  # Light background color
button_color = "#4CAF50"  # Green button color
button_text_color = "#FFFFFF"
label_color = "#333333"  # Dark text color

root.configure(bg=bg_color)

# Create and place widgets
title_label = tk.Label(root, text="Real-time Audio Source Distance Estimation", font=("Helvetica", 16), bg=bg_color, fg=label_color)
title_label.pack(pady=20)

instruction_label = tk.Label(root, text="Click 'Start Recording'", font=("Helvetica", 12), bg=bg_color, fg=label_color)
instruction_label.pack(pady=5)

start_button = tk.Button(root, text="Start Recording", command=start_recording, font=("Helvetica", 14), bg=button_color, fg=button_text_color)
start_button.pack(pady=20)

countdown_label = tk.Label(root, text="", font=("Helvetica", 12), bg=bg_color, fg=label_color)
countdown_label.pack(pady=10)

prediction_label = tk.Label(root, text="", font=("Helvetica", 14), bg=bg_color, fg=label_color)
prediction_label.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Helvetica", 12), bg="red", fg="white")
exit_button.pack(pady=10)

# Start the GUI main loop
root.mainloop()
