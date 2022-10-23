import matplotlib.pyplot as plt
import numpy as np


def plot_moving_avg(a):
    N = len(a)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = a[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg, label='running average')


def draw(stat_name, stats):
    for i in range(len(stats)):
        plt.title(stat_name + " for agent " + str(i + 1))
        plt.xlabel('episode')
        plt.ylabel(stat_name)
        plt.plot(stats[i], label=stat_name)
        plot_moving_avg(np.array(stats[i]))
        plt.legend()
        plt.show()

    means = []
    max_episodes = len(max(stats, key=len))
    for i in range(max_episodes):
        stat_sum = 0.
        stat_num = 0
        for stat in stats:
            if i < len(stat):
                stat_sum += stat[i]
                stat_num += 1
        means.append(stat_sum / stat_num)

    plt.title("Average " + stat_name + " for 9 agents")
    plt.xlabel('episode')
    plt.ylabel(stat_name)
    plt.plot(means, label=stat_name)
    plot_moving_avg(np.array(means))
    plt.legend()
    plt.show()