import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from pipeline import *

setup()

output = 'project_video_output.mp4'
input_clip = VideoFileClip('project_video.mp4')
output_clip = input_clip.fl_image(video_pipeline)
output_clip.write_videofile(output, audio=False)
