import matplotlib.animation as animation
from matplotlib.text import Text


class GifMaker:

    def __init__(self, mdp, title="", grid_size=1.5):
        self.fig, self.ax, self.base_image = mdp.visualise(gif=True, title=title, grid_size=grid_size)
        self.frames = [[self.base_image]]  # animations take in a list of lists of Artists
        self.frames = []

    def add_frame(self, frame, title=None):
        if title:
            title = [self.ax.text(0.5, 1.05, title, horizontalalignment='center', transform=self.ax.transAxes)]
            self.frames.append([self.base_image] + frame + title)
        else:
            self.frames.append([self.base_image] + frame)

    def save(self, filename):
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=500, blit=False,
                                        repeat_delay=2000)
        ani.save(filename)
