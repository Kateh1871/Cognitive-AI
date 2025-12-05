# mega plotting file
import matplotlib.pyplot as plt
import numpy as np


def GuessTrace(model_RNN, model_SERNN, task_idx = -1):
    # plots the target and model guess traces for 2 models

    fig, ax = plt.subplots(ncols=2)
    fig.set_figwidth(12)
    fig.tight_layout()
    hidden_choice_RNN = [np.argmax(x) for x in model_RNN.hidden_states[task_idx]['output']]
    hidden_choice_SERNN = [np.argmax(x) for x in model_SERNN.hidden_states[task_idx]['output']]

    ax[0].plot(model_SERNN.hidden_states[task_idx]['target'].cpu(), label='target')
    ax[0].plot(model_SERNN.hidden_states[task_idx]['output'], label='output')
    ax[0].legend()
    ax[0].set_ylabel('Direction Choice')
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_xlabel('Time')
    ax[0].set_title('SERNN model, 250 epochs, batch size 512, hidden size 8, \nperceptual decision making task')

    ax[1].plot(model_RNN.hidden_states[task_idx]['target'].cpu(), label='target')
    ax[1].plot(model_RNN.hidden_states[task_idx]['output'], label='output')
    ax[1].legend(loc='upper left')
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_title('RNN model, 250 epochs, batch size 512, hidden size 8, \nperceptual decision making task')
    ax[1].set_xlabel('Time')

    return fig, ax


def MapSERNN(model_SERNN):
    # plots the SERNN in physical space with connections
    # todo: sort out how i->j and j->i connections are shown
    fig, ax = plt.subplots(ncols=2)

    s_pos = model_SERNN.spatial.coords
    s_pos = np.array(s_pos)

    gamma_norm = lambda x, gamma:  x ** (1 / gamma)  # make very weak connections visible
    # gamma_norm = lambda x, gamma: x
    s_weights = list(model_SERNN.rnn.parameters())[2].detach().cpu().numpy()

    # i should stop writing lines like this, gets the highest absolute value of the array
    s_norm_value = np.max(s_weights) if np.max(s_weights)>np.abs(np.min(s_weights)) else np.abs(np.min(s_weights))
    s_weights = s_weights / s_norm_value  # normalise weights
    # the parameter order has changed?

    s_weights_upper = np.triu(s_weights)
    s_weights_lower = np.tril(s_weights)

    ax[0].scatter(s_pos[:, 0], s_pos[:, 1])
    for i, (x, y) in enumerate(s_pos):
        ax[0].text(x, y, str(i))

    for i in range(model_SERNN.hidden_size):
        for j in range(model_SERNN.hidden_size):
            x = [s_pos[i, 0], s_pos[j, 0]]
            y = [s_pos[i, 1], s_pos[j, 1]]

            colour = 'b' if np.abs(s_weights_upper[i, j]) == s_weights_upper[i, j] else 'o'

            ax[0].plot(x, y, alpha=gamma_norm(np.abs(s_weights_upper[i, j]), 4), color=colour)

    ax[1].scatter(s_pos[:, 0], s_pos[:, 1])
    for i, (x, y) in enumerate(s_pos):
        ax[1].text(x, y, str(i))
    for i in range(model_SERNN.hidden_size):
        for j in range(model_SERNN.hidden_size):
            x = [s_pos[i, 0], s_pos[j, 0]]
            y = [s_pos[i, 1], s_pos[j, 1]]

            colour = 'b' if np.abs(s_weights_lower[i, j]) == s_weights_lower[i, j] else 'o'

            ax[1].plot(x, y, alpha=gamma_norm(np.abs(s_weights_lower[i, j]), 4), color=colour)

    return fig, ax


def ActivityTrace(model_SERNN, model_RNN, task_idx = -1):
    # plots activity of hidden neurons for 2 models
    fig, ax = plt.subplots(nrows=2)

    s_activity = model_SERNN.hidden_states[task_idx]['hidden']
    s_activity = s_activity.detach().cpu().numpy()[:, 0, :]

    s_target = model_SERNN.hidden_states[task_idx]['target']
    s_target_idx = [i for i, j in enumerate(s_target) if j != 0]

    r_activity = model_RNN.hidden_states[task_idx]['hidden']
    r_activity = r_activity.detach().cpu().numpy()[:, 0, :]

    r_target = model_RNN.hidden_states[task_idx]['target']
    r_target_idx = [i for i, j in enumerate(r_target) if j != 0]

    ax[0].imshow(s_activity.T)
    ax[0].set_aspect('auto')
    ax[0].set_title('SERNN unit activity')
    xtwin = ax[0].twiny()
    xtwin.set_xticks(s_target_idx, labels=[])


    ax[1].imshow(r_activity.T)
    ax[1].set_aspect('auto')
    ax[1].set_title('RNN unit activity')
    ax[1].set_xlabel('Time')
    xtwin = ax[1].twiny()
    xtwin.set_xticks(r_target_idx, labels=[])

    fig.tight_layout()

    return fig, ax