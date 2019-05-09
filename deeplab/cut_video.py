#
# from moviepy.editor import VideoFileClip
# output2 = 'trucks2_cut.mp4'
# clip2 = VideoFileClip("DJI_0686.MOV").subclip(0,14)
#
# clip2.write_videofile(output2, audio=False)
# #clip2 = VideoFileClip("challenge_video.mp4").subclip(20,28)



from moviepy.editor import VideoFileClip
output2 = 'horse.mp4'
clip2 = VideoFileClip("horse_vid.mp4").subclip(120,130)

clip2.write_videofile(output2, audio=False)
#clip2 = VideoFileClip("challenge_video.mp4").subclip(20,28)
