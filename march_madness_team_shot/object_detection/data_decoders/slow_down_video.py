# Import everything needed to edit/save/watch video clips

import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import glob
import time
from moviepy.decorators import apply_to_mask, apply_to_audio


def speedx(clip, factor=None, final_duration=None):
    """
    Returns a clip playing the current clip but at a speed multiplied
    by ``factor``. Instead of factor one can indicate the desired
    ``final_duration`` of the clip, and the factor will be automatically
    computed.
    The same effect is applied to the clip's audio and mask if any.
    """

    if final_duration:
        factor = 1.0 * clip.duration / final_duration

    newclip = clip.fl_time(lambda t: factor * t, apply_to=['mask', 'audio'])

    if clip.duration is not None:
        newclip = newclip.set_duration(1.0 * clip.duration / factor)

    return newclip



white_output2 = '/home/priya/Documents/AI_Apps/soccer_project/soccer_mini_slow.mp4'
clip_in = VideoFileClip('/home/priya/Documents/AI_Apps/soccer_project/soccer_mini.mp4')
white_slow = speedx(clip_in, factor = 0.5)
white_slow.write_videofile(white_output2, audio=False)
