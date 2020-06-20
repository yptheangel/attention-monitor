#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import cv2
import json
import numpy as np
from zeromq.SerializingContext import SerializingContext

print("Connecting to hello world server…")
context = SerializingContext()
socket = context.socket(zmq.PUB)
socket.connect("tcp://localhost:5555")

def main():
    #  Socket to talk to server


    # #  Do 10 requests, waiting each time for a response
    # for request in range(10):
    #     print("Sending request %s …" % request)
    #     socket.send(b"Hello")
    #
    #     #  Get the reply.
    #     message = socket.recv()
    #     print("Received reply %s [ %s ]" % (request, message))

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        data={
            'id': 2,
            'data1': 1.2,
            'data2': 1.4
        }
        publish(frame, data)
        # print("Sending request …")
        # data = {
        #     'frame': frame
        # }
        # socket.send(json.dumps(data))

        cv2.imshow('client1', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


        # message = socket.recv()
        # print("Received reply [ %s ]" % (message))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def publish(image, data):
    if image.flags['C_CONTIGUOUS']:
        # if image is already contiguous in memory just send it
        socket.send_array(image, data, copy=False)
    else:
        # else make it contiguous before sending
        image = np.ascontiguousarray(image)
        socket.send_array(image, data, copy=False)

if __name__ == '__main__':
    main()