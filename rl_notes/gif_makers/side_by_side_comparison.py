import os
import imageio
import numpy as np
from gridworld import GridWorld
from gif_maker import GifMaker


def run_learner(mdp, learner, qfunction, learner_name, out_filename, grid_size=1.5, episodes=20):
    gridworld = GridWorld()
    gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)

    for episode in range(0, episodes + 1):
        learner.execute(episodes=1)
        title = "%s after episode %d" % (learner_name, episode)
        image_texts = gridworld.visualise_q_function(qfunction, title=title, grid_size=grid_size, gif=True)
        gif_maker.add_frame(image_texts, title=title)

    gif_maker.save(out_filename)

def join_gif(filename1, filename2, out_filename):

    #Create reader object for the gif
    gif1 = imageio.get_reader(filename1)
    gif2 = imageio.get_reader(filename2)

    #If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    #Create writer object
    new_gif = imageio.get_writer(out_filename)

    for frame_number in range(number_of_frames):
        if (frame_number < number_of_frames):
            img1 = gif1.get_data(frame_number)
            img2 = gif2.get_data(frame_number)
            new_image = np.hstack((img1, img2))
            new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()

    os.system("bash compress.bash %s" % (out_filename))
