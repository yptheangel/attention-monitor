# # -*- coding: utf-8 -*-
# import random
# from itertools import count
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import csv


# plt.style.use('fivethirtyeight')

# x_vals =[]
# y_vals =[]

# index = count()


# def data_writter_initialze(fieldname_list):
#     with open('data.csv', 'w') as csv_file:
#         csv_writer = csv.DictWriter(csv_file, fieldnames=fieldname_list)
#         csv_writer.writeheader()
        
        
# def data_writter_write(dict_data, fieldname_list):
#     with open('data.csv', 'a') as csv_file:
#         csv_writer = csv.DictWriter(csv_file, fieldnames=fieldname_list)
#         csv_writer.writerow(dict_data)
#         print(dict_data)        

# def animate(i):
#     data = pd.read_csv('data.csv')
#     timestamp = data['timestamp']
#     yaw = data['yaw']
#     pitch = data['pitch']
#     roll = data['roll']
#     plt.cla()
#     plt.plot(timestamp, yaw)
#     plt.plot(timestamp, pitch)
#     plt.plot(timestamp, roll)
    
#     plt.legend(loc='upper_left')
#     plt.tight_layout()
    
# ani = FuncAnimation(plt.gcf(), animate)

# #for analytics purpose
# #data = pd.read_csv('data.csv')
# #data.describe()
