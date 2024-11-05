import cv2
import numpy as np
from pyvda import VirtualDesktop

# Initialize video capture with the device (use 1 if it's the secondary camera)
cap = cv2.VideoCapture(1)

# Define a threshold for the amount of motion to notify the user
MOTION_THRESHOLD = 10000  # Adjust this based on your needs

# Read the first frame and convert it to grayscale
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
i = 0

while True:
    i += 1
    # Capture the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(prev_gray, gray)

    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total area of motion
    motion_area = sum(cv2.contourArea(c) for c in contours)

    # Check if the motion exceeds the defined threshold
    if motion_area > MOTION_THRESHOLD and i > 100:
        print("Motion detected! Area:", motion_area)
        # Switch to the next desktop (wrapping around if at the last one)
        try:
            current_desktop = VirtualDesktop.current()
            target_desktop = VirtualDesktop((current_desktop.number - 2))
            target_desktop.go()
        except:
            try:
                current_desktop = VirtualDesktop.current()
                target_desktop = VirtualDesktop((current_desktop.number - 1))
                target_desktop.go()
            except:
                print("We tried")

    # Update the previous frame
    prev_gray = gray

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
