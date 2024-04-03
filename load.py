import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Set up GPIO pins
G1_pin = 17
G2_pin = 18
G3_pin = 22
G4_pin = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(G1_pin, GPIO.OUT)
GPIO.setup(G2_pin, GPIO.OUT)
GPIO.setup(G3_pin, GPIO.OUT)
GPIO.setup(G4_pin, GPIO.OUT)

# Function to control pins based on peanut grade
def control_pins(grade):
  GPIO.output(G1_pin, GPIO.LOW)
  GPIO.output(G2_pin, GPIO.LOW)
  GPIO.output(G3_pin, GPIOLOW)
  GPIO.output(G4_pin, GPIO.LOW)

  if grade == "G1":
    GPIO.output(G1_pin, GPIO.HIGH)
  elif grade == "G2":
    GPIO.output(G2_pin, GPIO.HIGH)
  elif grade == "G3":
    GPIO.output(G3_pin, GPIO.HIGH)
  elif grade == "G4":
    GPIO.output(G4_pin, GPIO.HIGH)

# Function to classify peanut grade based on size and model prediction
def classify_peanut(size, model):
  # Preprocess size data (if needed)
  processed_size = np.array([size])  # Assuming size is a single value

  # Make prediction using the model
  prediction = model.predict(processed_size)
  predicted_grade = np.argmax(prediction)  # Get the grade with highest probability

  # Map the predicted class index to your grade labels (G1, G2, etc.)
  grade_mapping = {0: "G1", 1: "G2", 2: "G3", 3: "G4"}  # Example mapping dictionary
  grade = grade_mapping[predicted_grade]

  return grade

# Capture image from webcam
def capture_image():
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  cap.release()
  return frame

# Preprocess captured image
def preprocess_image(image):
  # Add preprocessing steps here if needed (e.g., resizing, normalization)
  return image

# Main function
def main():
  # Load the pre-trained model (replace with your model path)
  model = tf.keras.models.load_model('path/to/your/model.h5')

  while True:
    frame = capture_image()
    preprocessed_frame = preprocess_image(frame)

    # Use image processing techniques to detect and measure the peanut
    # Example: Use contour detection, size measurement, etc.
    # Here, let's assume peanut size is measured and stored in 'peanut_size'
    peanut_size = 50  # Example size value

    # Classify peanut grade based on size and model prediction
    grade = classify_peanut(peanut_size, model)

    # Control GPIO pins based on peanut grade
    control_pins(grade)

    time.sleep(1)  # Adjust delay as needed

if __name__ == "__main__":
  main()