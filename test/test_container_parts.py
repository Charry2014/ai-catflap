'''Pieces the container relies on: GPIO fallback, stream sources, and the camera stream server.'''
import io
import os
import subprocess
import sys
import threading
import time
import types

import cv2 as cv
import numpy as np
import pytest

import stubs  # noqa: F401
import catflapcontrol as cfc
from imgsrcstream import ImageSourceStream

ROOT = os.path.join(os.path.dirname(__file__), '..')


class FakeGPIO:
    BCM, OUT, LOW, HIGH = 0, 1, 0, 1

    def __init__(self, fail=False):
        self.fail, self.out, self.cleaned = fail, [], False

    def setmode(self, m):
        pass

    def setup(self, pin, mode):
        if self.fail:
            raise RuntimeError("no /dev/gpiomem")

    def output(self, pin, v):
        self.out.append(v)

    def input(self, pin):
        return self.out[-1]

    def cleanup(self):
        self.cleaned = True


def test_flap_is_simulated_without_gpio(monkeypatch):
    monkeypatch.setattr(cfc, 'GPIO', None)
    c = cfc.CatFlapControl()
    c.lock()
    assert c.locked and c.gpio_state == 1
    c.exit()
    assert not c.locked and c.gpio_state == 0


def test_flap_drives_the_relay(monkeypatch):
    gpio = FakeGPIO()
    monkeypatch.setattr(cfc, 'GPIO', gpio)
    c = cfc.CatFlapControl()
    c.lock()
    c.unlock()
    c.exit()
    assert gpio.out == [0, 1, 0, 0] and gpio.cleaned


def test_gpio_failure_falls_back_to_simulation_unless_hardware_is_required(monkeypatch):
    monkeypatch.setattr(cfc, 'GPIO', FakeGPIO(fail=True))
    monkeypatch.setattr(cfc, 'GPIO_MODE', 'auto')
    c = cfc.CatFlapControl()
    c.lock()
    assert c.locked
    monkeypatch.setattr(cfc, 'GPIO_MODE', 'hardware')
    with pytest.raises(RuntimeError):
        cfc.CatFlapControl()


def test_stream_source_selection():
    assert ImageSourceStream.can_supply_images("http://pi:8000/stream.mjpg")
    assert ImageSourceStream.can_supply_images("rtsp://cam/stream")
    assert not ImageSourceStream.can_supply_images("0")
    assert not ImageSourceStream.can_supply_images("/tmp/x.mp4")


def test_factory_picks_stream_for_urls():
    from imgsrcfactory import ImageSourceFactory
    assert isinstance(ImageSourceFactory.create_source("http://pi:8000/stream.mjpg"), ImageSourceStream)


class FlakyCapture:
    '''Opens fine, fails to read twice, then delivers frames'''
    instances = []

    def __init__(self, url):
        self.reads = 0
        FlakyCapture.instances.append(self)

    def isOpened(self):
        return True

    def read(self):
        self.reads += 1
        if len(FlakyCapture.instances) < 3:
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def test_stream_reconnects_after_interruption(monkeypatch):
    import imgsrcstream
    FlakyCapture.instances = []
    monkeypatch.setattr(imgsrcstream.cv2, 'VideoCapture', FlakyCapture)
    monkeypatch.setattr(ImageSourceStream, 'RECONNECT_DELAY', 0)
    src = ImageSourceStream(source="http://x/stream.mjpg")
    assert src.open() == 0
    assert src.get_image() is not None
    assert len(FlakyCapture.instances) == 3          # connected, then reconnected twice


def test_stream_gives_up(monkeypatch):
    import imgsrcstream

    class Dead(FlakyCapture):
        def read(self):
            return False, None

    monkeypatch.setattr(imgsrcstream.cv2, 'VideoCapture', Dead)
    monkeypatch.setattr(ImageSourceStream, 'RECONNECT_DELAY', 0)
    monkeypatch.setattr(ImageSourceStream, 'RECONNECT_TRIES', 3)
    src = ImageSourceStream(source="http://x/stream.mjpg")
    assert src.open() == 0
    assert src.get_image() is None and not src.isopen


def test_mjpeg_server_can_be_read_by_the_stream_source(monkeypatch):
    '''The container's only link to the Pi camera: OpenCV reading the server's multipart stream'''
    # A fake camera that feeds JPEGs to the encoder's output
    camera = types.ModuleType("picamera2")
    encoders = types.ModuleType("picamera2.encoders")
    outputs = types.ModuleType("picamera2.outputs")
    libcamera = types.ModuleType("libcamera")
    libcamera.Transform = lambda **kw: None
    stop = threading.Event()

    class Cam:
        def configure(self, c): pass
        def create_video_configuration(self, **kw): return kw

        def start_recording(self, encoder, output):
            def feed():
                n = 0
                while not stop.is_set():
                    img = np.full((120, 160, 3), (n * 10) % 255, np.uint8)
                    output.fileoutput.write(cv.imencode('.jpg', img)[1].tobytes())
                    n += 1
                    time.sleep(1 / 30)
            threading.Thread(target=feed, daemon=True).start()

        def stop_recording(self): stop.set()

    camera.Picamera2 = Cam
    encoders.MJPEGEncoder = lambda: None
    outputs.FileOutput = lambda o: types.SimpleNamespace(fileoutput=o)
    for name, mod in (("picamera2", camera), ("picamera2.encoders", encoders),
                      ("picamera2.outputs", outputs), ("libcamera", libcamera)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.path.insert(0, os.path.join(ROOT, 'docker'))
    try:
        import mjpeg_server
    finally:
        sys.path.pop(0)

    output = mjpeg_server.StreamingOutput()
    httpd = mjpeg_server.StreamingServer(('127.0.0.1', 0), mjpeg_server.make_handler(output))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    Cam().start_recording(None, types.SimpleNamespace(fileoutput=output))
    try:
        src = ImageSourceStream(source=f"http://127.0.0.1:{port}/stream.mjpg")
        assert src.open() == 0
        frames = [src.get_image() for _ in range(5)]
        assert all(f is not None and f.shape == (120, 160, 3) for f in frames)
        src.close()
    finally:
        stop.set()
        httpd.shutdown()


def test_entrypoint_builds_the_command_line(tmp_path):
    env = dict(os.environ, STREAM="http://h/s.mjpg", TRIGGER="1,2,3,4", MODEL="/m.tflite",
               WEBSITE="http://w", RECORD_VIDEO="1", RECORD_OVERLAYS="0", PREROLL="4", POSTROLL="",
               EXTRA_ARGS="--num_threads 2", DETECTOR="litert")
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\necho "ARGS: $*"\n')
    fake_python.chmod(0o755)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    out = subprocess.run(["sh", os.path.join(ROOT, 'docker', 'entrypoint.sh')], env=env,
                         capture_output=True, text=True, timeout=20).stdout
    line = [l for l in out.splitlines() if l.startswith("ARGS:")][0]
    assert "--stream http://h/s.mjpg" in line and "--web http://w" in line
    assert "--record_video" in line and "--record_overlays" not in line
    assert "--preroll 4" in line and "--postroll" not in line and line.endswith("--num_threads 2")
    assert "--headless" in line and "--detector litert" in line
