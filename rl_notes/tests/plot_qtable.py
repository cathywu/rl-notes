
import matplotlib.pyplot as plt

def plot_qtable(headers, data):

    fig, ax = plt.subplots()

    # Format cell data as text for consistent decimal place formatting
    formatted_data = []
    for row in data:
        formatted_row = []
        formatted_row.append(row[0])
        for cell in row[1:len(row)]:
            formatted_row.append("%.2f" % cell)
        formatted_data.append(formatted_row)
    table = ax.table(cellText=formatted_data, colLabels=headers, loc='center', cellLoc='right', edges='open')

    cells = table.get_celld()
    for i in range(0, len(headers)):
        cells[(0, i)].set_text_props(horizontalalignment='right', weight='bold', color='black')
        cells[(0, i)].visible_edges = 'BT'
        cells[(0, i)].visible_edges = 'BT'
        cells[(len(data), i)].visible_edges = 'B'

    table.set_fontsize(12)
    table.scale(0.7, 1.5)
    ax.axis('off')

    plt.show()


headers=["State", "Up", "Down", "Right", "Left"]
data = [[(0,0), 0.00, 0.00, 0.00, 0.00],
        [(0,1), 0.00, 0.00, 0.00, 0.00],
        [(0,2), 0.00, 0.00, 0.00, 0.00],
        [(1,0), 0.00, 0.00, 0.00, 0.00],
        [(1,1), 0.00, 0.00, 0.00, 0.00],
        [(1,2), 0.00, 0.00, 0.00, 0.00],
        [(2,0), 0.00, 0.00, 0.00, 0.00],
        [(2,1), 0.00, 0.00, 0.00, 0.00],
        [(2,2), 0.00, 0.00, 0.00, 0.00],
        [(3,0), 0.00, 0.00, 0.00, 0.00],
        [(3,1), 0.00, 0.00, 0.00, 0.00],
        [(3,2), 0.00, 0.00, 0.00, 0.00]] 
plot_qtable(headers, data)


data = [[(0, 0), 0.50, 0.42, 0.39, 0.42],
        [(0, 1), 0.56, 0.44, 0.51, 0.51],
        [(0, 2), 0.58, 0.51, 0.63, 0.57],
        [(1, 0), 0.09, 0.18, 0.06, 0.43],
        [(1, 1), 0.00, 0.00, 0.00, 0.00],
        [(1, 2), 0.64, 0.65, 0.74, 0.59],
        [(2, 0), 0.41, 0.00, 0.00, 0.00],
        [(2, 1), 0.69, 0.09, -0.24, 0.24],
        [(2, 2), 0.73, 0.61, 0.85, 0.65],
        [(3, 0), -0.02, 0.00, 0.00, 0.00],
        [(3, 1), 0.00, 0.00, 0.00, 0.00],
        [(3, 2), 0.00, 0.00, 0.00, 0.00]]
plot_qtable(headers, data)
