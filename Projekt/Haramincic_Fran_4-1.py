import socket
import threading
import struct as s
from queue import Queue
import math as m
from flask import Flask, jsonify, render_template
import json

HOST = "192.168.127.131" # VM ip address
# HOST = "192.168.40.50" # Robot ip address
PORT = 30003

current_data = {
    'tool_position' : {'x': 0, 'y': 0, 'z': 0},
    'tool_orientation' : {'rx': 0, 'ry': 0, 'rz': 0},
    'joint_positions' : [0]*6,
    'motor_temperatures' : [0]*6,
    'tool_forces' : {'x': 0, 'y': 0, 'z': 0},
    'controller_timestamp' : 0,
    'connection_status' : 'Disconnected'
}

data_lock = threading.Lock()
data_queue = Queue()
app = Flask(__name__)

def run_client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        print(f"Client connected to --> {HOST}:{PORT}")
        try:
            while True:
                data_pack = client.recv(1500)
                data_queue.put(data_pack)
        except ConnectionError as e:
            print(f"Connection error: {e}")
            with data_lock:
                current_data['connection_status'] = 'Disconnected'

def parse_data():
    while True:
        try:
            data = data_queue.get()
            tool_positions = [pos*1000 for pos in s.unpack_from('!3d', data, 588)]
            tool_orientations = s.unpack_from('!3d', data, 612)
            joints = [m.degrees(rad) for rad in s.unpack_from('!6d', data, 252)]
            temps = s.unpack_from('!6d', data, 692)
            forces = s.unpack_from('!3d', data, 540)
            timestamp = s.unpack_from('!d', data, 4)

            with data_lock:
                current_data['tool_position'] = {
                    'x': tool_positions[0],
                    'y': tool_positions[1],
                    'z': tool_positions[2]
                }
                current_data['tool_orientation'] = {
                    'rx': tool_orientations[0],
                    'ry': tool_orientations[1],
                    'rz': tool_orientations[2]
                }
                current_data['joint_positions'] = list(joints)
                current_data['motor_temperatures'] = list(temps)
                current_data['tool_forces'] = {
                    'x': forces[0],
                    'y': forces[1],
                    'z': forces[2]
                }
                current_data['controller_timestamp'] = timestamp
                current_data['connection_status'] = 'Connected'
            
            print(current_data)
        except Exception as e:
            print(f"Error while parsing data: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    with data_lock:
        return jsonify(current_data)

if __name__ == "__main__":
    t_client = threading.Thread(target=run_client, daemon=True)
    t_parsing = threading.Thread(target=parse_data, daemon=True)
    t_flask = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=50000, debug=False), daemon=True)

    t_client.start()
    t_parsing.start()
    t_flask.start()

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nExiting...")