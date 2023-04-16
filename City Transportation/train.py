from Environment import CityEnv
from Model import BusDriver
import numpy as np
import matplotlib.pyplot as plt

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Average Waiting Time', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for l in lines:
            ax2.axhline(y=l, color='r', linestyle='-')

    plt.savefig(filename, dpi=1200)

if __name__ == '__main__':
    env = CityEnv()
    agent = BusDriver(gamma=0.99, epsilon=1.0, lr=0.001)
    scores = []
    eps_history = []
    n_runs = 10000
    version = 4

    for i in range(n_runs):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(-score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(f'episode {i}, average waiting time {-score}, average {avg_score}, epsilon {agent.epsilon}')

    x = [i+1 for i in range(n_runs)]
    plot_file = 'plots/BusDriver_v' + str(version) + '.png'
    plotLearning(x, scores, eps_history, plot_file)

    model_file = 'models/BusDriver_v' + str(version) + '.pth'
    agent.save_model(model_file)

    env_file = 'envs/BusDriver_v' + str(version) + '.pkl'
    env.save(env_file)