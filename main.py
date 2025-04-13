from pyvda import VirtualDesktop
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import argparse
import time
import signal
import logging
import paho.mqtt.client as mqtt

# MQTT Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Motion detection script with desktop switching and masking.")
    parser.add_argument('--threshold', type=int, default=4000, help='Threshold for motion area detection.')
    parser.add_argument('--camera', type=int, help='Single camera ID to use.')
    parser.add_argument('--cameras', type=int, nargs='+', help='List of multiple camera IDs to use.')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit.')
    parser.add_argument('--masks', type=str, nargs='+', help='List of mask images corresponding to the cameras.')
    parser.add_argument('--reactivation', type=int, default=5, help='Time in seconds before reactivating a disabled camera.')
    parser.add_argument('--grace-frames', type=int, default=100, help='Number of frames for a grace period after reactivation.')
    parser.add_argument('--mqtt-broker', type=str, default=None, help='MQTT broker address.')
    parser.add_argument('--mqtt-port', type=int, default=None, help='MQTT broker port.')
    parser.add_argument('--mqtt-topic', type=str, default=None, help='MQTT topic to publish events.')
    parser.add_argument('--mqtt-client-id', type=str, default="motion_detector", help='MQTT client ID.')
    parser.add_argument('--mqtt-username', type=str, default=None, help='MQTT username for authentication.')
    parser.add_argument('--mqtt-password', type=str, default=None, help='MQTT password for authentication.')
    return parser.parse_args()


args = parse_args()

# MQTT Configuration
MQTT_BROKER = args.mqtt_broker
MQTT_PORT = args.mqtt_port
MQTT_TOPIC = args.mqtt_topic
MQTT_CLIENT_ID = args.mqtt_client_id

mqtt_client = None
if MQTT_BROKER and MQTT_PORT and MQTT_TOPIC:
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    if args.mqtt_username and args.mqtt_password:
        mqtt_client.username_pw_set(args.mqtt_username, args.mqtt_password)
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()


def list_available_cameras():
    try:
        graph = FilterGraph()
        cameras = graph.get_input_devices()
        print("Available cameras:")
        for i, cam in enumerate(cameras):
            print(f"Camera {i}: {cam}")
    except Exception as e:
        print(f"Error listing cameras: {e}")
    exit()


def initialize_cameras(camera_ids):
    caps = {}
    prev_frames = {}
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            logging.warning(f"Unable to access camera {cam_id}. Skipping...")
            cap.release()
            continue
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_frames[cam_id] = gray_frame
            caps[cam_id] = cap
            logging.info(f"Camera {cam_id} initialized successfully.")
        else:
            logging.warning(f"Camera {cam_id} is accessible but returned no frames.")
            cap.release()
    return caps, prev_frames


def load_masks(camera_ids, mask_paths, caps):
    masks = {}
    if mask_paths:
        if len(mask_paths) != len(camera_ids):
            raise ValueError("Number of masks must match the number of cameras.")
        for cam_id, mask_path in zip(camera_ids, mask_paths):
            if cam_id not in caps:
                continue  # Skip missing camera
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Unable to load mask image: {mask_path}")
                # Resize the mask to match the camera's frame size
                ret, frame = caps[cam_id].read()
                if ret:
                    h, w = frame.shape[:2]
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                    masks[cam_id] = mask_bin
                    logging.info(f"Loaded and resized mask for Camera {cam_id} from {mask_path}")
                else:
                    logging.warning(f"Unable to retrieve frame for Camera {cam_id} to scale mask.")
            except Exception as e:
                logging.warning(f"Failed to process mask for Camera {cam_id}: {e}")
    return masks


def trigger_alarm():
    try:
        if mqtt_client and MQTT_TOPIC:
            mqtt_client.publish(MQTT_TOPIC, payload="tripped", qos=1, retain=False)
            logging.info(f"MQTT event published to topic '{MQTT_TOPIC}': 'tripped'")
        else:
            logging.info("MQTT client or topic not specified. Taking no action.")

    except Exception as e:
        logging.error(f"Failed to notify MQTT broker: {e}")


def process_motion_detection(caps, prev_frames, motion_threshold, reactivation_time, grace_frames, masks):
    disabled_cameras = {}
    frame_counters = {cam_id: 0 for cam_id in caps}  # Track frame counters per camera to enforce grace period
    while True:
        for cam_id, cap in list(caps.items()):
            if cam_id in disabled_cameras and time.time() < disabled_cameras[cam_id]:
                continue  # Skip this camera if it's still disabled
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Camera {cam_id} failed to provide a frame. Skipping...")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Increment the frame counter for this camera to implement the reactivation grace period
            frame_counters[cam_id] += 1
            if frame_counters[cam_id] <= grace_frames:
                logging.debug(f"Camera {cam_id}: Grace period in effect ({frame_counters[cam_id]}/{grace_frames} frames).")
                prev_frames[cam_id] = gray
                continue

            # Calculate difference between consecutive frames
            diff = cv2.absdiff(prev_frames[cam_id], gray)

            # Apply mask if it exists for this camera
            if cam_id in masks:
                diff = cv2.bitwise_and(diff, diff, mask=masks[cam_id])

            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate total motion area
            motion_area = sum(cv2.contourArea(c) for c in contours)
            logging.debug(f"Camera {cam_id}: Detected motion area {motion_area}")
            if motion_area > motion_threshold:
                logging.info(f"Motion detected on Camera {cam_id}! Area: {motion_area}")
                trigger_alarm()
                logging.info(f"Disabling Camera {cam_id} temporarily.")
                cap.release()
                del caps[cam_id]
                disabled_cameras[cam_id] = time.time() + reactivation_time  # Schedule reactivation
                del frame_counters[cam_id]  # Remove its frame counter (will be reset on reactivation)
            prev_frames[cam_id] = gray

        # Reactivate cameras as their cooldown expires
        for cam_id in list(disabled_cameras.keys()):
            if time.time() >= disabled_cameras[cam_id]:
                new_cap = cv2.VideoCapture(cam_id)
                if new_cap.isOpened():
                    ret, frame = new_cap.read()
                    if ret:
                        prev_frames[cam_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        caps[cam_id] = new_cap
                        frame_counters[cam_id] = 0  # Reset frame counter for grace period
                        del disabled_cameras[cam_id]
                        logging.info(f"Reactivated Camera {cam_id}. Grace period of {grace_frames} frames started.")
                    else:
                        new_cap.release()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            logging.info("Exiting motion detection loop.")
            break


def cleanup(caps):
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    logging.info("Released all camera resources.")


def handle_exit(signum, frame):
    logging.info("Signal received. Disconnecting MQTT client and exiting gracefully...")
    mqtt_client.loop_stop()  # Stop MQTT loop
    mqtt_client.disconnect()  # Cleanly disconnect the client
    exit(0)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    # Handle interrupt signals for a graceful cleanup
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    args = parse_args()
    if args.list_cameras:
        list_available_cameras()
    motion_threshold = args.threshold
    reactivation_time = args.reactivation
    grace_frames = args.grace_frames
    camera_ids = args.cameras if args.cameras else [args.camera] if args.camera is not None else []
    if not camera_ids:
        logging.error("No cameras specified. Use --camera or --cameras to select cameras.")
        return
    logging.info("Initializing cameras...")
    caps, prev_frames = initialize_cameras(camera_ids)
    if not caps:
        logging.error("No valid cameras available. Exiting.")
        return

    masks = load_masks(camera_ids, args.masks, caps)

    try:
        process_motion_detection(caps, prev_frames, motion_threshold, reactivation_time, grace_frames, masks)
    finally:
        cleanup(caps)


if __name__ == "__main__":
    main()