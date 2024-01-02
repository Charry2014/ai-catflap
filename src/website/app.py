from flask import Flask, render_template
from flask_socketio import SocketIO
from collections import deque
import base64

app = Flask(__name__)
socketio = SocketIO(app)
log_file_name = None

# Initialize deque to store the last 20 lines
log_buffer = deque(maxlen=20)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log')
def index_log():
    # Get initial log data
    log_data = '\n'.join(log_buffer)
    return render_template('index_log.html', log_data=log_data)

@app.route('/stream')
def index_image():
    return render_template('index_image.html')


@socketio.on('update_list')
def update_list():
    # Get the latest log data
    log_data = get_last_lines_from_log(20)
    socketio.emit('new_element', {'element': log_data})

def get_last_lines_from_log(num_lines):
    try:
        # Read the entire log file and split it into lines
        with open(log_file_name, 'r') as log_file:
            lines = log_file.readlines()

        # Update the log buffer with the latest lines
        log_buffer.extend(lines[-num_lines:])

        # Return the latest log data
        return ''.join(lines[-num_lines:])
    except Exception as e:
        print(f"Error reading log file: {e}")
        return "Error reading log file"


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image_data')
def handle_image(data):
    try:
        # Save the image to a file (optional)
        # Decode base64 image data
        # image_data = base64.b64decode(data['image'])
        # with open('received_image.png', 'wb') as f:
        #    f.write(image_data)

        # Broadcast the image to all clients
        socketio.emit('display_image', {'image': data['image']})

    except Exception as e:
        print(f"Error handling image data: {e}")


if __name__ == '__main__':
    # Find a log file to tail - this is hacky, but when testing the log file is in one place
    # and in the Docker container it is in another (the /log location actually)
    from os import access, R_OK
    from os.path import isfile
    locations = ['/log/catcam.log', './example.log']
    for l in locations:
        if isfile(l) and access(l, R_OK):
            log_file_name = l
    assert log_file_name != None, print("Error: Could not find a log file to tail")

    # Go, go, go!
    socketio.run(app, host='0.0.0.0', debug=True)
