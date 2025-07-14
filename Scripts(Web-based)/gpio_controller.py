# gpio_controller.py
try:
    import RPi.GPIO as GPIO
    real_gpio = True
except (ImportError, RuntimeError):
    # Use mock GPIO for non-Raspberry Pi systems
    real_gpio = False

    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0

        @staticmethod
        def setmode(mode):
            print(f"[MOCK] Set mode: {mode}")

        @staticmethod
        def setup(pin, mode):
            print(f"[MOCK] Setup pin {pin} as {mode}")

        @staticmethod
        def output(pin, state):
            print(f"[MOCK] Pin {pin} set to {'HIGH' if state else 'LOW'}")

        @staticmethod
        def cleanup():
            print("[MOCK] GPIO cleanup")

    GPIO = MockGPIO()

# Define the pin number you're using
DEVICE_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(DEVICE_PIN, GPIO.OUT)

def turn_on_device():
    GPIO.output(DEVICE_PIN, GPIO.HIGH)

def turn_off_device():
    GPIO.output(DEVICE_PIN, GPIO.LOW)

def cleanup_gpio():
    GPIO.cleanup()
