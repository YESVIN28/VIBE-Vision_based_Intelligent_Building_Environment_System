# sensor_server.py
import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
from flask import Flask, jsonify, request
import threading
import socket
import sys

# GPIO setup
LED_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Start with LED off

# DHT Sensor setup
dht_device = adafruit_dht.DHT11(board.D4)

# Shared state
person_count = 0
person_count_lock = threading.Lock()
current_temp = 0
light_intensity = 0

# Add these global variables
led_status = False

# In-memory store for UI-added devices (simulate persistent storage)
ui_devices = {}

# Flask app
app = Flask(__name__)

@app.route('/api/device/add', methods=['POST'])
def add_device():
    data = request.json
    device_type = data.get('type')
    device_name = data.get('name')
    device_room = data.get('room')
    if not all([device_type, device_name, device_room]):
        return jsonify({"status": "error", "message": "Missing fields"}), 400
    import time
    device_id = f"{device_type}-{int(time.time())}"
    ui_devices[device_id] = {
        "type": device_type,
        "name": device_name,
        "room": device_room,
        "state": False,
        "online": False,
        "brightness": 0 if device_type in ["light", "led"] else None,
        "speed": 0 if device_type == "fan" else None,
        "volume": 0 if device_type == "tv" else None,
        "temperature": 20 if device_type == "ac" else None
    }
    return jsonify({"status": "success", "device_id": device_id})

@app.route('/api/devices', methods=['GET'])
def get_device_status():
    # Check DHT11 status
    try:
        temp = dht_device.temperature
        dht_online = temp is not None
    except Exception:
        temp = None
        dht_online = False

    # LED is online if GPIO is set up
    led_online = True

    # Start with hardware devices
    devices = {
        "led": {
            "type": "light",
            "name": "Smart LED",
            "state": led_status,
            "online": led_online,
            "brightness": light_intensity
        },
        "dht11": {
            "type": "sensor",
            "name": "DHT11 Sensor",
            "state": dht_online,
            "online": dht_online,
            "temperature": temp if temp is not None else 0
        }
    }

    # Add UI-added devices, mark as offline if not real hardware
    for dev_id, dev in ui_devices.items():
        if dev_id not in devices:
            devices[dev_id] = {
                **dev,
                "state": False,
                "online": False
            }

    return jsonify({
        "status": "success",
        "devices": devices
    })

@app.route('/api/device/toggle', methods=['POST'])
def toggle_led():
    global led_status, light_intensity
    data = request.json
    if data.get('device') == 'led':
        led_status = bool(data.get('state'))
        GPIO.output(LED_PIN, GPIO.HIGH if led_status else GPIO.LOW)
        if not led_status:
            light_intensity = 0
        elif light_intensity == 0:
            light_intensity = 100
        return jsonify({"status": "success", "state": led_status, "brightness": light_intensity})
    return jsonify({"status": "error", "message": "Invalid device"}), 400

@app.route('/api/device/control', methods=['POST'])
def control_led():
    global led_status, light_intensity
    data = request.json
    if data.get('device') == 'led' and data.get('control') == 'brightness':
        value = int(data.get('value', 0))
        light_intensity = max(0, min(100, value))
        led_status = light_intensity > 0
        GPIO.output(LED_PIN, GPIO.HIGH if led_status else GPIO.LOW)
        return jsonify({"status": "success", "brightness": light_intensity, "state": led_status})
    return jsonify({"status": "error", "message": "Invalid control"}), 400

def get_person_count():
    with person_count_lock:
        return person_count

def set_person_count(value):
    global person_count
    with person_count_lock:
        person_count = value

def update_devices():
    global current_temp, light_intensity, led_status
    while True:
        try:
            pc = get_person_count()
            if pc == 0:
                GPIO.output(LED_PIN, GPIO.LOW)
                led_status = False
                current_temp = 0
                light_intensity = 0
                print("LED OFF (no person detected)")
            else:
                temp = dht_device.temperature
                current_temp = temp if temp is not None else 0
                light_intensity = min(100, pc * 25)
                led_status = True
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("LED ON (person detected)")
        except Exception as e:
            print(f"DHT error: {e}")
        time.sleep(2)

def start_socket_server():
    host = '0.0.0.0'
    port = 5001
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"Listening for person count on {host}:{port}...")
    while True:
        data, _ = sock.recvfrom(1024)
        try:
            value = int(data.decode().strip())
            set_person_count(value)
            print(f"Updated person count: {value}")
        except ValueError:
            print(f"Invalid data: {data}")

def cleanup():
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("GPIO cleaned up.")

if __name__ == "__main__":
    try:
        threading.Thread(target=update_devices, daemon=True).start()
        threading.Thread(target=start_socket_server, daemon=True).start()
        app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cleanup()
        sys.exit(0)