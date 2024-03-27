import matplotlib.pyplot as plt

from gridworld import GridWorld

def draw_action_sequence(initial_state, actions, offset=0.0, gap=0.2, head_width=0.05):
    mdp = GridWorld()
    fig = mdp.visualise()
    ax =  fig.get_axes()[0]

    state = initial_state
    i = 1
    for action in actions:
        arrow_params = {'color': 'black', 'shape': 'full', 'head_width': head_width}
        if action == GridWorld.UP:
            next_state = (state[0], state[1] + 1)
            ax.arrow(state[0] + offset, state[1] + gap, 0, (1 - head_width) - 2 * gap, **arrow_params)
            ax.text(state[0] + 0.5*gap, state[1] + 1.5*gap, str(i))
        elif action == GridWorld.DOWN:
            next_state = (state[0], state[1] - 1)
            ax.arrow(state[0] - offset, state[1] - gap, 0, -(1 - head_width) + 2 * gap, **arrow_params)
            ax.text(state[0] - 0.75*gap, state[1] - 1.5*gap, str(i))
        elif action == GridWorld.RIGHT:
            next_state = (state[0] + 1, state[1])
            ax.arrow(state[0] + gap, state[1] + offset, (1 - head_width) - 2 * gap, 0, **arrow_params)
            ax.text(state[0] + 1.5*gap, state[1] + 0.55*gap, str(i))
        elif action == GridWorld.LEFT:
            next_state = (state[0] - 1, state[1])
            ax.arrow(state[0] - gap, state[1] - offset, -(1 - head_width) + 2 * gap, 0, **arrow_params)
            ax.text(state[0] - 0.75*gap, state[1] - 1.5*gap, str(i))

        state = next_state
        i += 1


