'''Replace modules that only exist on the Pi, so the state machine imports anywhere.'''
import sys
import types

if "RPi.GPIO" not in sys.modules:
    gpio = types.SimpleNamespace(BCM=0, OUT=0, LOW=0, HIGH=1, setmode=lambda *a: None,
                                 setup=lambda *a: None, output=lambda *a: None,
                                 input=lambda *a: 0, cleanup=lambda: None)
    rpi = types.ModuleType("RPi")
    rpi.GPIO = gpio
    sys.modules["RPi"] = rpi
    sys.modules["RPi.GPIO"] = gpio

# Other things that are not installed everywhere
for name in ("picamera2", "libcamera", "socketio"):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.Picamera2 = type("Picamera2", (), {"set_logging": staticmethod(lambda *a: None), "INFO": 0})
        module.Transform = lambda **kw: None
        module.Client = type("Client", (), {})
        sys.modules[name] = module
