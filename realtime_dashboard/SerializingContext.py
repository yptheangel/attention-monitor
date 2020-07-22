import zmq
import numpy as np

class SerializingSocket(zmq.Socket):
    """Numpy array serialization methods.
    Modelled on PyZMQ serialization examples.
    Used for sending / receiving OpenCV images, which are Numpy arrays.
    Also used for sending / receiving jpg compressed OpenCV images.
    """

    def send_array(self, A, data=None, flags=0, copy=True, track=False):
        md = dict(
            data=data,
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):

        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        return (md['data'], A.reshape(md['shape']))


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket