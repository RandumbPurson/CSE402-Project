import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def inRange(rmin, rmax, *args):
    return sum([not (rmin <= val < rmax) for val in args]) == 0

class IConfusionMatrixDisplay:
    """Add a popup image display to a confusion matrix plot.
        - Click to cycle through images of the class pair
    """
    def __init__(self, fig, ax, img_matrix):
        """
        Params:
        - fig: figure object of the CM plot
        - ax: axis object of the CM plot
        - img_matrix: a 2D list of lists, where each element is a list of images. Should be organized so that `img_matrix[true][pred]` accesses the list containing images of the class `true` with predicted class `pred`
        """
        self.fig = fig
        self.n_classes = len(img_matrix)
        self.fig_width, self.fig_height = self.fig.get_size_inches()*self.fig.dpi
        x0, y0, w, h = ax.get_position().bounds
        self.x0, w = x0*self.fig_width, w*self.fig_width
        self.y0, h = y0*self.fig_height, h*self.fig_height
        
        self.cell_w = w/self.n_classes
        self.cell_h = h/self.n_classes

        # Selected
        self.img_xy = (None, None)
        self.img = None
        self.img_idx = -1
        self.im = OffsetImage(np.zeros((160, 160, 3)), zoom=0.5)
        self.img_matrix = img_matrix

        # https://github.com/choosehappy/Snippets/blob/master/interactive_image_popup_on_hover.py
        self.xybox=(50., 50.)
        self.ab = AnnotationBbox(
            self.im, (0,0), xycoords='figure pixels', 
            xybox=self.xybox, boxcoords="offset points",  
            pad=0.3, arrowprops=dict(arrowstyle="->")
        )
        # add it to the axes and make it invisible
        ax.add_artist(self.ab)
        self.ab.set_visible(False)

        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.hover)

    def getCellIndex(self, x, y):
        cell_x = ((x - self.x0) // self.cell_w).astype(int)
        cell_y = (self.n_classes - 1 - (y - self.y0) // self.cell_h).astype(int)
        return cell_x, cell_y

    def getBox(self, x, y):
        ws = (x > self.fig_width/2.)*-1 + (x <= self.fig_width/2.) 
        hs = (y > self.fig_height/2.)*-1 + (y <= self.fig_height/2.)
        return (self.xybox[0]*ws, self.xybox[1]*hs)

    def hover(self, event):
        cell_x, cell_y = self.getCellIndex(event.x, event.y)
        if not inRange(0, self.n_classes, cell_x, cell_y):
            return

        if self.img is None or (cell_x, cell_y) != self.img_xy or (cell_x, cell_y) == (None, None):
            self.ab.set_visible(False)
            return
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        self.ab.xybox = self.getBox(event.x, event.y)
        # make annotation box visible
        self.ab.set_visible(True)
        # place it at the position of the hovered scatter point
        self.ab.xy =(event.x, event.y)
        # set the image corresponding to that point
        # im.set_data(self.img) #if the dataset is too large to load into memory, can instead replace this command with a realtime load
        self.fig.canvas.draw_idle()

    def _reset_img(self):
        self.img = None
        self.img_xy = (None, None)
        self.img_idx = -1

    def click(self, event):
        cell_x, cell_y = self.getCellIndex(event.x, event.y)
        if not inRange(0, self.n_classes, cell_x, cell_y):
            self._reset_img()
            return

        n_choices = len(self.img_matrix[cell_y][cell_x])
        if n_choices == 0:
            self.ab.set_visible(False)
            self._reset_img()
            return

        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        self.ab.xybox = self.getBox(event.x, event.y)
        # make annotation box visible
        self.ab.set_visible(True)
        # place it at the position of the hovered scatter point
        self.ab.xy =(event.x, event.y)
        # set the image corresponding to that point
        self.img_xy = (cell_x, cell_y)
        self.img_idx = (self.img_idx + 1) % n_choices
        self.img = self.img_matrix[cell_y][cell_x][self.img_idx]
        self.im.set_data(self.img) #if the dataset is too large to load into memory, can instead replace this command with a realtime load
        self.fig.canvas.draw_idle()
