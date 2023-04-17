from matplotlib import pyplot as plt
from Model import PortfolioManager
from Environment import TradingEnvironment
import numpy as np

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i+1 for i in range(N)]
    
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, running_avg)
    plt.savefig(filename, dpi=1200)


if __name__ == '__main__':
    env = TradingEnvironment()
    agent = PortfolioManager(actor_lr=0.00001, critic_lr=0.00005, tau=0.001)
    scores = []
    n_runs = 10000
    version = 1

    # agent.load_models()
    # # load score history
    # score_history = np.load('score_history.npy', allow_pickle=True).tolist()

    np.random.seed(0)

    score_history = []
    for i in range(n_runs):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            new_state, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)

        if i % 25 == 0:
            agent.save_models()
            np.save('score_history.npy', score_history)

        print(f'episode {i} score {score} average score {np.mean(score_history[-100:])}')

    agent.save_models()

    score_history = np.array(score_history)
    score_history = score_history[score_history < np.percentile(score_history, 95)]
    score_history = score_history[score_history > np.percentile(score_history, 5)]

    plot_file = 'plots/PortfolioManager_v' + str(version) + '.png'
    
    plotLearning(score_history, plot_file)
    