from tempfile import NamedTemporaryFile
from matplotlib import pyplot as plt
from IPython.display import HTML
from matplotlib import animation

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=None,#fps=20, 
                      extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def show_animation(*args, **kwargs):
    """ Creates and shows animation """
    anim = animation.FuncAnimation(*args, **kwargs)
    return display_animation(anim)

def head_signals_tight_movie(fig, data, title):
    fig_title = fig.suptitle(title, fontsize=16)
    # initialization function: plot the background of each frame
    def init():
        changed_lines = []
        for ax in fig.axes:
            for line in ax.lines[:-1]:
                line.set_data(range(30), [0] * 30)
                changed_lines.append(line)
        return changed_lines

    # animation function.  This is called sequentially
    def animate(frame_data):
        i_frame, frame_pattern = frame_data
        changed_parts = []
        for i_sensor, ax in enumerate(fig.axes):
            if frame_pattern.ndim == 2 or frame_pattern.shape[2] == 1:
                ax.lines[0].set_data(range(len(frame_pattern[i_sensor])),
                    frame_pattern[i_sensor])
                changed_parts.append(ax.lines[0])
            else:
                for i_line in xrange(frame_pattern.shape[2]):
                    ax.lines[i_line].set_data(range(len(frame_pattern[i_sensor])),
                        frame_pattern[i_sensor,:,i_line])
                    changed_parts.append(ax.lines[i_line])
                    
        fig_title = fig.texts[0]
        fig_title.set_text("{:s} {:d}".format(title, i_frame + 1))
        changed_parts.append(fig_title)
        return changed_parts

    return show_animation(fig, animate, init_func=None,
                                   frames=list(enumerate(data)),
                                   interval=1000, blit=True)