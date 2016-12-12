
# coding: utf-8

# In[1]:

from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=1000./anim._interval, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)


# In[2]:

from IPython.display import HTML
from matplotlib import pyplot as plt

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


# In[3]:

from matplotlib import animation
import numpy as np

def display_images(frames, interval=100):
    ''' frames -- a list/array of np.array images. Plays all frames in the notebook as a clip.'''
    frames = [np.atleast_3d(f) for f in frames]
    if frames[0].shape[2]!=3:
        frames = [np.dstack((f,f,f)) for f in frames]

    print(frames[0].shape)
    fig = plt.figure()
    img = plt.imshow(frames[0])
    
    def init():
        img.set_data(np.random.random((frames[0].shape[1], frames[0].shape[2])))
        return [img]

    # animation function.  This is called sequentially
    def animate(i):
        img.set_array(frames[i])
        return [img]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    return display_animation(animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames), interval=interval, blit=True))


# In[ ]:




# In[ ]:




# In[ ]:



