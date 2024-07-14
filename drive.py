import socketio
from flask import Flask, render_template
from train import model, data_aug
from io import BytesIO
import base64
import eventlet
import eventlet.wsgi
import numpy 


app = Flask(__name__)
sio = socketio.Server()
target_speed = 20
shape = (100, 100, 3)
model = model(True, shape)

@sio.on('telemetry')
def telemetry(sid, data):
    img = data["image"]
    speed = float(data["speed"])

    throttle = 1 - (speed / target_speed)

    img_bytes = BytesIO(base64.b64decode(img))
    image, _ = data_aug(img_bytes, None, False)

    str_angle = model.predict(numpy.array([image]))[0][0]

    print(str_angle, throttle)
    control(str_angle, throttle)

@sio.on('connect')
def connect(sid, environment):
    print("connect", sid)
    control(0, 0)

def control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    