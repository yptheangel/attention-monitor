import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys
import time
import threading
from queue import Queue

class MetricsMonitor(object):
    def __init__(self):

        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Attention Monitor')
        self.win.setWindowTitle('Attention Monitor')

        # set size of output window
        self.win.setGeometry(5, 115, 1280, 720)

        ratios_xlabels = [(0, '0'), (60, '60'), (120, '120')]
        ratios_xaxis = pg.AxisItem(orientation='bottom')
        ratios_xaxis.setTicks([ratios_xlabels])

        ratios_ylabels = [(0, '0'), (0.25,'0.25'), (0.5, '0.5'), (0.75,'0.75'), (1.0, '1.0')]
        ratios_yaxis = pg.AxisItem(orientation='left')
        ratios_yaxis.setTicks([ratios_ylabels])

        poses_xlabels = [(0, '0'), (60, '60'), (120, '120')]
        poses_xaxis = pg.AxisItem(orientation='bottom')
        poses_xaxis.setTicks([poses_xlabels])

        poses_ylabels = [(-45, '-45'),(0, '0'), (45,'45')]
        poses_yaxis = pg.AxisItem(orientation='left')
        poses_yaxis.setTicks([poses_ylabels])

        self.ratios = self.win.addPlot(
            title='Face Metrics', row=1, col=1, axisItems={'bottom': ratios_xaxis, 'left': ratios_yaxis},
        )
        self.poses = self.win.addPlot(
            title='Head Pose Metrics', row=2, col=1, axisItems={'bottom': poses_xaxis, 'left': poses_yaxis},
        )

        self.lgd = pg.LegendItem((80,60), offset=(70,20))
        self.lgd.setParentItem(self.ratios.graphicsItem())
        self.c1 = self.ratios.plot([], pen='c', name='mar')
        self.c2 = self.ratios.plot([], pen='m', name='ear')
        self.lgd.addItem(self.c1, 'Mouth Aspect Ratio')
        self.lgd.addItem(self.c2, 'Eye Aspect Ratio')

        self.lgd2 = pg.LegendItem((80,60), offset=(70,20))
        self.lgd2.setParentItem(self.poses.graphicsItem())
        self.c3 = self.poses.plot([], pen='r', name='yaw')
        self.c4 = self.poses.plot([], pen='g', name='pitch')
        self.c5 = self.poses.plot([], pen='b', name='roll')

        self.lgd2.addItem(self.c3, 'Yaw')
        self.lgd2.addItem(self.c4, 'Pitch')
        self.lgd2.addItem(self.c5, 'Roll')

        self.x = np.arange(0, 2 * 60, 2)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'mar':
                self.traces[name] = self.ratios.plot(pen='c', width=3,fillLevel=0,fillBrush=(0,255,255,80))
                self.ratios.setYRange(0, 1, padding=0)
                self.ratios.setXRange(0, 2 * 60, padding=0.005)
            elif name == 'ear':
                self.traces[name] = self.ratios.plot(pen='m', width=3)
                self.ratios.setYRange(0, 1, padding=0)
                self.ratios.setXRange(0, 2 * 60, padding=0.005)
            elif name == 'yaw':
                self.traces[name] = self.poses.plot(pen='r', width=3,fillLevel=0,fillBrush=(255,0,0,80))
                self.poses.setYRange(-45, 45, padding=0)
                self.poses.setXRange(0, 2 * 60, padding=0.005)
            elif name == 'pitch':
                self.traces[name] = self.poses.plot(pen='g', width=3)
                self.poses.setYRange(-45, 45, padding=0)
                self.poses.setXRange(0, 2 * 60, padding=0.005)
            elif name == 'roll':
                self.traces[name] = self.poses.plot(pen='b', width=3)
                self.poses.setYRange(-45, 45, padding=0)
                self.poses.setXRange(0, 2 * 60, padding=0.005)

    def stream(self, q1,q2,q3,q4,q5):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5

    def update(self):
        if not self.q1.empty() and not self.q2.empty():
            self.mar = self.q1.get()
            self.ear = self.q2.get()
            self.set_plotdata(name='mar', data_x=self.x, data_y=self.mar,)
            self.set_plotdata(name='ear', data_x=self.x, data_y=self.ear,)

        elif not self.q3.empty() and not self.q4.empty() and not self.q5.empty():
            self.yaw = self.q3.get()
            self.pitch = self.q4.get()
            self.roll = self.q5.get()
            self.set_plotdata(name='yaw', data_x=self.x, data_y=self.yaw,)
            self.set_plotdata(name='pitch', data_x=self.x, data_y=self.pitch,)
            self.set_plotdata(name='roll', data_x=self.x, data_y=self.roll,)

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(30)
        self.start()

if __name__ == '__main__':
    monitor_app = MetricsMonitor()
    monitor_app.animation()
