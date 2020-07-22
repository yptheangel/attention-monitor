import threading
from queue import Queue
from plotter import MetricsMonitor
import numpy as np

# modify my class to become a customized thread
# class User(object):
class User(threading.Thread):

    def __init__(self, userid):
        # super(User,self).__init(userid=userid)
        # initialize the queue with a dict of 5 streams
        self.userid = userid
        self.q = Queue()
        # stream1 = np.zeros(60)
        # stream2 = np.zeros(60)
        # stream3 = np.zeros(60)
        # stream4 = np.zeros(60)
        # stream5 = np.zeros(60)
        stream1 = np.ones(60)
        stream2 = np.ones(60)
        stream3 = np.ones(60)
        stream4 = np.ones(60)
        stream5 = np.ones(60)
        self.dict = {"mar_stream": stream1,
                     "ear_stream": stream2,
                     "yaw_stream": stream3,
                     "pitch_stream": stream4,
                     "roll_stream": stream5,
                     }
        self.q.put(self.dict)

    #     How to update the values in a while loop way??
    def update_values(self, datarecord):
        self.datarecord =  datarecord
        if (self.datarecord['id'] == self.userid):
            self.dict = self.q.get()
            self.dict["mar_stream"] = np.append(self.dict["mar_stream"][1:], datarecord['record']['mar'])
            self.dict["ear_stream"] = np.append(self.dict["ear_stream"][1:], datarecord['record']['ear'])
            self.dict["yaw_stream"] = np.append(self.dict["yaw_stream"][1:], datarecord['record']['yaw'])
            self.dict["pitch_stream"] = np.append(self.dict["pitch_stream"][1:], datarecord['record']['pitch'])
            self.dict["roll_stream"] = np.append(self.dict["roll_stream"][1:], datarecord['record']['roll'])
            self.q.put(self.dict)
            print("put")

    def getName(self):
        print(self.userid)

    def run(self):
        self.monitor_app = MetricsMonitor(self.userid)
        self.monitor_app.stream(self.q)
        self.monitor_app.animation()

    # def monitor(self):
    #     monitor_app = MetricsMonitor()
    #     monitor_app.stream(self.q)
    #     monitor_app.animation()
