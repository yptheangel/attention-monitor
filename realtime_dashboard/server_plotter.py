import zmq
import cv2
from SerializingContext import SerializingContext
import numpy as np

context = SerializingContext()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
# socket.bind("tcp://10.10.10.163:5555")
socket.bind("tcp://*:5555")
from threading import Thread
from queue import Queue
from plotter import MetricsMonitor

def server(q):
    #
    #   Hello World server in Python
    #   Binds REP socket to tcp://*:5555
    #   Expects b"Hello" from client, replies with b"World"
    #
    clients=[]
    this_dict = q.get()

    while True:
        #  Wait for next request from client
        data, image = subscribe()
        # client_id = data['id']
        # print(data)
        # print(client_id)
        # print(client_id.dtype)
        if data['record'] != None:
            this_dict["mar_stream"] = np.append(this_dict["mar_stream"][1:], data['record']['mar'])
            this_dict["ear_stream"] = np.append(this_dict["ear_stream"][1:], data['record']['ear'])
            this_dict["yaw_stream"] = np.append(this_dict["yaw_stream"][1:], data['record']['yaw'])
            this_dict["pitch_stream"] = np.append(this_dict["pitch_stream"][1:], data['record']['pitch'])
            this_dict["roll_stream"] = np.append(this_dict["roll_stream"][1:], data['record']['roll'])
            q.put(this_dict)
            # print("put")

            # print(data['record']['yawn_count'])
            # print(data['record']['blink_count'])
            # print(data['record']['lost_focus_duration'])
            # print(data['record']['face_not_present_duration'])

            # sample data {'id': '100', 'sortKey': '3cd72332-b2d9-11ea-b115-0d30b3991a0b', 'timestamp': 1592645668.660121, 'yaw': -7.538459777832031, 'pitch': -4.917228698730469, 'roll': 1.390106201171875, 'ear': 0.33526643780010046, 'blink_count': 6, 'mar': 0.02564102564102564, 'yawn_count': 0, 'lost_focus_count': 1, 'lost_focus_duration': 1.3774120807647705, 'face_not_present_duration': 0.34656667709350586}
        cv2.imshow(data['id'],  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # 1 window for each RPi
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

def monitor(q):
    monitor_app = MetricsMonitor(q)
    monitor_app.animation()

if __name__ == '__main__':
    Q = Queue()
    stream1 = np.zeros(60)
    stream2 = np.zeros(60)
    stream3 = np.zeros(60)
    stream4 = np.zeros(60)
    stream5 = np.zeros(60)
    dict = {"mar_stream" : stream1,
            "ear_stream": stream2,
            "yaw_stream": stream3,
            "pitch_stream": stream4,
            "roll_stream": stream5,
            }
    Q.put(dict)
    t1 = Thread(name='Server Thread', target=server, args=(Q,))
    t1.start()
    monitor(Q)
    t1.join()
