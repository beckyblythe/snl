#matplotlib.use("Agg")
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1000)


def plot_animated (data, node, saddle, sep_slope, cycle_boundary):

    fig1 = plt.figure()
        
    l, = plt.plot([], [], 'r-')
    plt.xlim((-70,0))
    plt.ylim((-.05,.7))
    plt.xlabel('x')
    plt.title('test')
    plt.plot(node[0], node[1],marker='o', color='r')
    plt.plot(cycle_boundary[0],cycle_boundary[1], marker = 'o')
    plt.plot(saddle[0], saddle[1], marker = 'o', color = 'm')
    y = np.linspace(-.1,.7,50)
    x = sep_slope[0]/sep_slope[1]*(y-saddle[1])+saddle[0]
    plt.plot(saddle[0], saddle[1], color = 'm')
    plt.plot(x,y, color = 'm')
    line_ani = animation.FuncAnimation(fig1, update_line, data[0].shape[0], fargs=(data, l),
                                       interval=.00005)
    plt.show()
    line_ani.save('2.mp4', writer=writer)
