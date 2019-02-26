import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("plot", help="name of plot")
args = parser.parse_args()

font = {'size': 40}
matplotlib.rc('font', **font)

version1_dir = "version1"
version1_data = {
        "EDSS": "2019-02-20-12-43-36",
        "Handover": "2019-02-21-15-35-31",
        "Nearest": "2019-02-21-15-37-26",
        "Random": "2019-02-21-15-40-49"
}

speed1_dir = "speed1"
speed1_data = {
        "EDSS": "2019-02-26-01-38-14",
        "Handover": "2019-02-25-13-43-29",
        "Nearest": "2019-02-25-13-46-24",
        "Random": "2019-02-25-14-00-19"
}


def read_data(directory, agent, date, phase, measure):
    return np.array(
        pd.read_csv(
            "{directory}/run_{agent}_{date}_{phase}-tag-{measure}.csv".format(directory=directory,
                                                                              agent=agent,
                                                                              date=date,
                                                                              phase=phase,
                                                                              measure=measure)
        )["Value"]
    )


def linear_filter(value_list, window_size=3):
    def shift_sublist(values, shift):
        shifted_list = np.roll(values, shift)
        if shift > 0:
            shifted_list[:shift] = values[:shift]
        else:
            shifted_list[shift:] = values[shift:]
        return shifted_list

    filtered_list = np.copy(value_list)
    for s in range(-window_size, window_size+1):
        if s != 0:
            filtered_list += shift_sublist(value_list, s)
    return filtered_list / (2 * window_size + 1)


def exponential_moving_average(value_list, window_size=10):
    ema_list = [value_list[0]]
    multiplier = 2 / (window_size + 1)
    for i in range(1, len(value_list)):
        ema_list.append(value_list[i]*multiplier + ema_list[-1]*(1-multiplier))
    return ema_list


def set_axis_range(y_axis_range, x_rate=100, y_rate=0.5):
    # Axis
    plt.xticks(np.arange(0, 1001, x_rate))
    plt.yticks(np.arange(y_axis_range[0], y_axis_range[1] + 1, y_rate))
    plt.ylim(y_axis_range)
    plt.grid(color="white")


def get_case_properties(agent):
    if "EDSS (train)" == agent:
        color = "firebrick"
        marker = "o"
        name = "EDMS (train)"
    elif "EDSS (test)" == agent or "EDSS" == agent:
        color = "orangered"
        marker = "D"
        name = "EDMS (test)"
    elif "Handover" == agent:
        color = "forestgreen"
        marker = "v"
        name = "Greedy (handover)"
    elif "Nearest" == agent:
        color = "steelblue"
        marker = "^"
        name = "Greedy (nearest)"
    else:
        color = "gray"
        marker = ","
        name = "Random"
    return {
        "color": color,
        "marker": marker,
        "label": name
    }


def plot_reward(directory_name, data_dict):
    # Average reward over simulations
    x_axis = np.array(range(1, 1001))

    data = {
        "EDSS (test)": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_OverallScore_mean_1"),
        "EDSS (train)": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "train", "Reward_OverallScore_mean_1"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_OverallScore_mean_1"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_OverallScore_mean_1"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_OverallScore_mean_1")
    }

    for agent in data:
        plt.plot(x_axis, exponential_moving_average(data[agent], 100), markevery=50, markersize=20, linewidth=2, **get_case_properties(agent))

    # 0 line
    plt.axhline(0, color="black", linestyle="dotted")

    # set_axis_range([-3.5, 1], x_rate=100, y_rate=0.5)
    plt.grid(True)

    # Label
    plt.xlabel("Simulation")
    plt.ylabel("Average Reward")

    # Legend
    plt.legend(facecolor="white", loc=4)

    for agent in data:
        plt.scatter(x_axis, data[agent], alpha=0.1, s=100, **get_case_properties(agent))

    plt.show()


def plot_statistics(directory_name, data_dict):
    reward_mean = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_OverallScore_mean_1"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_OverallScore_mean_1"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_OverallScore_mean_1"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_OverallScore_mean_1")
    }

    effectiveness_mean = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_Effectiveness_mean_1"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_Effectiveness_mean_1"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_Effectiveness_mean_1"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_Effectiveness_mean_1")
    }

    penalty_mean = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_Penalty_mean_1"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_Penalty_mean_1"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_Penalty_mean_1"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_Penalty_mean_1")
    }

    reward_stddev = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_OverallScore_stddev"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_OverallScore_stddev"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_OverallScore_stddev"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_OverallScore_stddev")
    }

    effectiveness_stddev = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_Effectiveness_stddev"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_Effectiveness_stddev"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_Effectiveness_stddev"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_Effectiveness_stddev")
    }

    penalty_stddev = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Reward_Penalty_stddev"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Reward_Penalty_stddev"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Reward_Penalty_stddev"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Reward_Penalty_stddev")
    }

    fig, axes = plt.subplots(2, 3, sharex='all')

    # Reward mean
    axes[0, 0].set_title("Reward")
    #axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_ylabel("Average")
    axes[0, 0].boxplot([reward_mean[agent] for agent in reward_mean], notch=True)
    axes[0, 0].set_xticklabels([agent for agent in reward_mean])
    axes[0, 0].yaxis.grid(True)

    # Effectiveness mean
    axes[0, 1].set_title("Effectiveness")
    #axes[0, 1].set_ylabel('Average Effectiveness')
    axes[0, 1].boxplot([effectiveness_mean[agent] for agent in effectiveness_mean], notch=True)
    axes[0, 1].set_xticklabels([agent for agent in effectiveness_mean])
    axes[0, 1].yaxis.grid(True)

    # Penalty mean
    axes[0, 2].set_title("Penalty")
    #axes[0, 2].set_ylabel('Average Penalty')
    axes[0, 2].boxplot([penalty_mean[agent] for agent in penalty_mean], notch=True)
    axes[0, 2].set_xticklabels([agent for agent in penalty_mean])
    axes[0, 2].yaxis.grid(True)

    # Reward stddev
    #axes[1, 0].set_ylabel('Reward Stddev')
    axes[1, 0].set_ylabel("Standard deviation")
    axes[1, 0].boxplot([reward_stddev[agent] for agent in reward_stddev], notch=True)
    axes[1, 0].set_xticklabels([agent for agent in reward_stddev], rotation=45)
    axes[1, 0].yaxis.grid(True)

    # Effectiveness stddev
    #axes[1, 1].set_ylabel('Effectiveness Stddev')
    axes[1, 1].boxplot([effectiveness_stddev[agent] for agent in effectiveness_stddev], notch=True)
    axes[1, 1].set_xticklabels([agent for agent in effectiveness_stddev], rotation=45)
    axes[1, 1].yaxis.grid(True)

    # Penalty stddev
    #axes[1, 2].set_ylabel('Penalty Stddev')
    axes[1, 2].boxplot([penalty_stddev[agent] for agent in penalty_stddev], notch=True)
    axes[1, 2].set_xticklabels([agent for agent in penalty_stddev], rotation=45)
    axes[1, 2].yaxis.grid(True)

    plt.show()


def plot_execution_time(directory_name, data_dict):
    execution_time_mean = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Summary_ExecutionTime_mean_1"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Summary_ExecutionTime_mean_1"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Summary_ExecutionTime_mean_1"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Summary_ExecutionTime_mean_1")
    }

    execution_time_stddev = {
        "EDSS": read_data(directory_name, "EDSS(DQN)", data_dict["EDSS"], "test", "Summary_ExecutionTime_stddev"),
        "Handover": read_data(directory_name, "NoHandover", data_dict["Handover"], "test", "Summary_ExecutionTime_stddev"),
        "Nearest": read_data(directory_name, "Nearest", data_dict["Nearest"], "test", "Summary_ExecutionTime_stddev"),
        "Random": read_data(directory_name, "Random", data_dict["Random"], "test", "Summary_ExecutionTime_stddev")
    }

    fig, axes = plt.subplots(1, 2, sharex='all')

    axes[0].boxplot([execution_time_mean[agent] for agent in execution_time_mean], notch=True)
    axes[0].set_ylabel("Average (sec)")
    #axes[0].set_title("Execution Time")
    axes[0].set_xticklabels([agent for agent in execution_time_mean], rotation=45)
    axes[0].yaxis.grid(True)

    axes[1].boxplot([execution_time_stddev[agent] for agent in execution_time_stddev], notch=True)
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_xticklabels([agent for agent in execution_time_mean], rotation=45)
    axes[1].yaxis.grid(True)

    """
    fig, ax = plt.subplots()

    #ax.set_xlabel('Agent')
    #ax.set_ylabel('Average Execution Time')

    ax.boxplot([execution_time_mean[agent] for agent in execution_time_mean], notch=True)
    ax.set_xticklabels([agent for agent in execution_time_mean])
    ax.yaxis.grid(True)
    #axes[0].set_title("Greedy (nearest)")
    #axes[1].boxplot(data["Handover"])
    #axes[1].set_title("Greedy (handover)")
    #axes[2].boxplot(data["EDSS (train)"])
    #axes[2].set_title("EDSS (train)")
    #axes[3].boxplot(data["EDSS (test)"])
    #axes[3].set_title("EDSS (test)")
    """

    # Label
    #plt.title("Execution Time")
    #plt.ylabel("Average Execution Time (sec)")

    # Legend
    #plt.legend(facecolor="white")

    plt.show()


directory = speed1_dir
data = speed1_data


if args.plot == "reward":
    plot_reward(directory, data)
elif args.plot == "statistics":
    plot_statistics(directory, data)
elif args.plot == "execution_time":
    plot_execution_time(directory, data)

