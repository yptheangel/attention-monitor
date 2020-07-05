import zmq
import cv2
# from zeromq.SerializingContext import SerializingContext
from SerializingContext import SerializingContext


context = SerializingContext()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
# socket.bind("tcp://10.10.10.163:5555")
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
        # client_id = data['id']
        # print(data)
        # print(client_id)
        # print(client_id.dtype)
        if data['record'] != None:
            print(data)
            print(type(data))
            print(data['id'])
            print(data['record']['yaw'])
            print(data['record']['pitch'])
            print(data['record']['roll'])
            print(data['record']['ear'])
            print(data['record']['mar'])
            print(data['record']['yawn_count'])
            print(data['record']['blink_count'])
            print(data['record']['lost_focus_duration'])
            print(data['record']['face_not_present_duration'])

            # sample data {'id': '100', 'sortKey': '3cd72332-b2d9-11ea-b115-0d30b3991a0b', 'timestamp': 1592645668.660121, 'yaw': -7.538459777832031, 'pitch': -4.917228698730469, 'roll': 1.390106201171875, 'ear': 0.33526643780010046, 'blink_count': 6, 'mar': 0.02564102564102564, 'yawn_count': 0, 'lost_focus_count': 1, 'lost_focus_duration': 1.3774120807647705, 'face_not_present_duration': 0.34656667709350586}
            # put your pyqtchart here (wilson choo)

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

if __name__ == '__main__':
    main()