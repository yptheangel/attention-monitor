import time
import zmq
import json
import cv2
from zeromq.SerializingContext import SerializingContext

context = SerializingContext()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
socket.bind("tcp://*:5555")

def main():
    #
    #   Hello World server in Python
    #   Binds REP socket to tcp://*:5555
    #   Expects b"Hello" from client, replies with b"World"
    #
    clients=[]
    while True:
        #  Wait for next request from client
        data, image = subscribe()
        client_id = data['id']
        print(data)
        print(client_id)
        # print(client_id.dtype)

        cv2.imshow(str(client_id),  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # 1 window for each RPi
        cv2.waitKey(1)
        # image_hub.send_reply(b'OK')

def subscribe(copy=False):
    """Receives OpenCV image and text msg.
    Arguments:
      copy: (optional) zmq copy flag.
    Returns:
      msg: text msg, often the image name.
      image: OpenCV image.
    """

    msg, image = socket.recv_array(copy=False)
    return msg, image

if __name__ == '__main__':
    main()