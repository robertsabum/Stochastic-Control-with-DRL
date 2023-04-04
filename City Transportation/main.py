from Environment import CityEnv
from Model import BusDriver
import numpy as np
import matplotlib.pyplot as plt

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for l in lines:
            ax2.axhline(y=l, color='r', linestyle='-')

    plt.savefig(filename)

if __name__ == '__main__':
    env = CityEnv()
    agent = BusDriver()
    scores = []
    eps_history = []
    n_runs = 100

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
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_runs)]
    filename = 'bus_driver_performance.png'
    plotLearning(x, scores, eps_history, filename)