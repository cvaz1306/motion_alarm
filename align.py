import cv2
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Blend webcam feed with a static image.")
parser.add_argument('--camera', type=int, default=0, help="Camera index (default: 0)")
parser.add_argument('--image', type=str, required=True, help="Path to the image to blend with the camera feed")
args = parser.parse_args()

# Load the static image and convert it to grayscale
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Unable to load image {args.image}")
    exit()

# Capture the webcam feed
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("Error: Could not access the webcam")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize the grayscale image to match the camera frame size
    frame_resized = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize to the same size and blend the two images with a t=0.5 factor
    blended_frame = cv2.addWeighted(frame_gray, 0.5, frame_resized, 0.5, 0)

    # Show the blended frame
    cv2.imshow('Blended Webcam Feed', blended_frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
