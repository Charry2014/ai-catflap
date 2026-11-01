#!/usr/bin/python3
"""Serves the Pi camera as an MJPEG stream over http, for the container to read.

Run it on the Pi itself (not in the container), with the system Python that has
picamera2:   python3 docker/mjpeg_server.py
The container then reads http://host.docker.internal:8000/stream.mjpg, which is
the compose default. See docker/catflap-camera.service to start it at boot.
"""
import argparse
import io
import logging
import socketserver
from http import server
from threading import Condition

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput


class StreamingOutput(io.BufferedIOBase):
    '''The encoder writes one JPEG per call; clients wait for the next one.'''
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


def make_handler(output):
    class StreamingHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/stream.mjpg':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:      # The client went away
                logging.info('Client %s gone: %s', self.client_address, e)

        def log_message(self, *args):
            pass
    return StreamingHandler


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--no-flip', action='store_true', help="Don't rotate the image 180 degrees")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    picam2 = Picamera2()
    frame_us = int(1_000_000 / args.fps)
    picam2.configure(picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        transform=Transform(hflip=0 if args.no_flip else 1, vflip=0 if args.no_flip else 1),
        controls={"FrameDurationLimits": (frame_us, frame_us)}))
    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    try:
        StreamingServer(('', args.port), make_handler(output)).serve_forever()
    finally:
        picam2.stop_recording()


if __name__ == '__main__':
    main()
