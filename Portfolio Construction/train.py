from matplotlib import pyplot as plt
from Model import PortfolioManager
from Environment import TradingEnvironment
import numpy as np

def plotLearning(returns, risks, filename):
    N = len(returns)
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(range(N), risks, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Annualized Volatility", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")


    ax2.plot(range(N), returns, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('10 year Returns', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")


    plt.title(f'Portfolio Manager performance over {N} episodes')

    plt.savefig(filename, dpi=1200)


if __name__ == '__main__':
    env = TradingEnvironment()
    agent = PortfolioManager(actor_lr=0.00001, critic_lr=0.00005, tau=0.001)
    scores = []
    returns = []
    risks = []
    n_runs = 10000
    version = 1

    agent.load_models()

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
        returns.append(env.net_return)
        risks.append(env.risk)

        if i % 25 == 0:
            agent.save_models()

        print(f'episode {i} score {score} average score {np.mean(score_history[-100:])}')

    agent.save_models()

    score_history = np.array(score_history)
    score_history = score_history[score_history < np.percentile(score_history, 95)]
    score_history = score_history[score_history > np.percentile(score_history, 5)]

    plot_file = 'plots/PortfolioManager_v' + str(version) + '.png'
    
    plotLearning(returns, risks, plot_file)
    