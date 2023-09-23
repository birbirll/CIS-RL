import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')


def simulate(policy, env, num_sim=50000):
    '''
    Simulate the environment with the given policy,
    the policy must be a discrete table.

    Return the average cost
    '''
    print(f"[ Simulating for {num_sim} episodes... ]")

    total_cost = 0

    # Simulate the environment for num_sim times
    for _ in range(num_sim):
        # Reset the environment
        x_t, _ = env.reset()
        cost, done, t = 0, False, 0
        # Iterate through all time steps
        while not done:
            # Take the action given by the policy
            u_t = policy[x_t][t]
            # Take a step in the environment
            (x_t, _), step_cost, done = env.step(u_t)
            cost += step_cost
            t += 1
        # Add the cost
        total_cost += cost

    return total_cost / num_sim


def simulate_lunarlander(agent, env, num_sim=50000):
    '''
    Simulate the environment with agent trained on lunarlander

    Return the average reward
    '''
    print(f"[ Simulating for {num_sim} episodes... ]")

    total_reward = 0

    # Simulate the environment for num_sim times
    for _ in range(num_sim):
        done = False
        state, info = env.reset()
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, truncated, info = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, reward, state_, done)
            state = state_

    return total_reward / num_sim





def plot_scores(scores, labels, title):
    '''
    Plot a bar chart for comparing the scores
    '''
    # Make the graph width adaptive to the number of scores
    width = len(scores)
    # Set the figure size
    plt.figure(figsize=(5, 5))
    # Set colors
    colors = ["Salmon", "SkyBlue", "SeaGreen", "Tomato", "SlateBlue", "Orange", "MediumSeaGreen"]
    # Set the x positions
    x_pos = np.arange(len(labels))
    # Plot the bar chart
    plt.bar(x_pos, scores, color=colors)
    # Show number on top of each bar
    for i, v in enumerate(scores):
        plt.text(i, v, str(int(v)), ha="center", va="bottom")
    # Set the labels
    plt.xticks(x_pos, labels)
    # Set the title
    plt.title(title)
    # Show the plot
    plt.show()

def plot_scores_with_runtime(scores, labels, title, runtime, figsize=(6, 3)):
    '''
    Plot a bar chart for comparing the scores as bar chart
    with runtime as line chart
    '''
    # Make the graph width adaptive to the number of scores
    width = len(scores)
    # Set the figure size
    plt.figure(figsize=figsize)
    # Set colors
    colors = ["Salmon", "SkyBlue", "SeaGreen", "Tomato", "SlateBlue", "Orange", "MediumSeaGreen"]
    # Set the x positions
    x_pos = np.arange(len(labels))
    # Plot the bar chart
    plt.bar(x_pos, scores, color=colors)
    # Show number on top of each bar
    for i, v in enumerate(scores):
        plt.text(i, v, str(int(v)), ha="center", va="bottom")
    # Set the labels
    plt.xticks(x_pos, labels)
    # Set the title
    plt.title(title)
    # Set the second y axis
    ax2 = plt.twinx()
    ax2.plot(x_pos, runtime, color="black", marker="o", linestyle="dashed")
    ax2.set_ylabel("Runtime (s)")
    # Show runtime on top of each point
    for i, v in enumerate(runtime):
        ax2.text(i, v, str(round(v, 2)), ha="center", va="bottom")
    # Show the plot
    plt.show()

    


def plot_learning_curve(scores, n_avg, title, xlabel, ylabel, figsize=(6, 3)):
    '''
    Plot the learning curve
    '''
    plt.figure(figsize=figsize)
    # Smooth the episode cost by taking the average of every n episodes
    episode_cost_smooth = np.convolve(scores, np.ones(n_avg), 'valid') / n_avg

    plt.plot(episode_cost_smooth, label='Episode score')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_learning_curve_with_baseline(scores, baseline, baseline_label, n_avg,
                                      title, xlabel, ylabel, figsize=(6, 3)):
    '''
    Plot the learning curve with a baseline
    '''
    plt.figure(figsize=figsize)
    # Smooth the episode cost by taking the average of every n episodes
    episode_cost_smooth = np.convolve(scores, np.ones(n_avg), 'valid') / n_avg

    plt.plot(episode_cost_smooth, label='Episode score')

    # plot the line for the baseline
    plt.plot([baseline] * len(episode_cost_smooth), label=baseline_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_state_visited_count(self, visit_count, figsize, title):
    '''
    Plot the number of times each state is visited as a heatmap
    '''
    plt.figure(figsize=figsize)
    # Convert table into log scale
    # count = np.log(self.visit_count + 1)
    count = visit_count
    # Transpose the table
    count = count.T

    # make the heatmap wider
    im = plt.imshow(np.repeat(count, 10, axis=1), cmap='Oranges', interpolation='nearest', 
                    norm=LogNorm())

    # Reverse the y-axis
    plt.gca().invert_yaxis()


    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.colorbar(im, label='Number of times visited', shrink=0.8)
    plt.title(title)
    
    plt.show()


if __name__ == "__main__":
    plot_scores([4416, 3717], ["A", "B"], "Test")
    