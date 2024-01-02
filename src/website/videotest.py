import cv2
import base64
import socketio
import time

# Create a SocketIO client
sio = socketio.Client()

# Connect to the Flask-SocketIO server
sio.connect('http://localhost:5000')  # Adjust the URL to match your server

# Video file path
video_file_path = '/Users/toby/work/projects/catflap/data/incoming/Cat-with-mouse/20230310/20221014-011509_catcam.mp4'

# Function to send frames to the server
def send_frames():
    cap = cv2.VideoCapture(video_file_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        jpeg_bytes = base64.b64encode(buffer.tobytes()).decode('utf-8')

        # print("Emitting image data " + jpeg_bytes)
        # Send the frame to the server
        sio.emit('image_data', {'image': jpeg_bytes})

        # Delay to simulate real-time streaming (adjust as needed)
        time.sleep(0.1)

    cap.release()

# Event handler for disconnect event
@sio.event
def disconnect():
    print('Disconnected from server')

if __name__ == '__main__':
    try:
        send_frames()
    except KeyboardInterrupt:
        print('User interrupted the script')
    finally:
        # Disconnect from the server
        sio.disconnect()
