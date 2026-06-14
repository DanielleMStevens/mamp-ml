import matplotlib.pyplot as plt
import numpy as np
import os

def plot_regression(ax, winding, breakpoints, slope, colors=[]):
    """
    ax: matplotlib axis
        Axis on which to plot the regression
    winding: ndarray(n)
        The winding number at each residue
    breakpoints: ndarray(int)
        Residue locations of the breakpoints
    slope: ndarray(n)
            Estimated slope in each winding segment
    colors: list of [(r, g, b) * (n_breakpoints+1)] (optional)
        List of colors to go with each interval
        If left blank, the default color cycler is used, starting at orange
    """
    boundaries = [0] + breakpoints.tolist() + [len(winding)]
    n_intervals = len(breakpoints)+1
    if len(colors) == 0:
        colors = [f"C{i+1}" for i in range(n_intervals)]
    elif len(colors) != n_intervals:
        raise ValueError("Must provide same number of colors as intervals")

    ax.plot(winding, c='C0', linewidth=1, zorder=100)
    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        linear = (i % 2) * slope * (np.arange(a, b) - (a + b - 1) / 2)
        y = linear + np.mean(winding[a:b])
        ax.plot(np.arange(a, b), y, c=colors[i], linestyle='--', linewidth=3)
    for b in breakpoints:
        ax.axvline(b, linestyle='--', c='k')
    ax.set_title('Piecewise linear regression on winding number graph')
    ax.set_xlabel('Residue number')
    ax.set_ylabel('Winding number')

def plot_residue_annotations_3d(X, breakpoints, colors=[], fac=10):
    """
    Use myavi to plot labels of the annotations in 3D

    Parameters
    ----------
    X: ndarray(n_residues, 3)
        Coordinates of the residues in 3D
    breakpoints: ndarray(int)
        Residue locations of the breakpoints
    colors: list of [(r, g, b) * (n_breakpoints+1)] (optional)
        List of colors to go with each interval, formatted as (r, g, b) in range [0, 255]
        If left blank, the default color cycler is used, starting at orange
    fac: int
        Interpolation factor for spline smoothing of the residue curve
    """
    from scipy.interpolate import CubicSpline
    from mayavi import mlab

    ## Step 1: Figure out labels for regions
    N = X.shape[0]
    regions = breakpoints.tolist() + [N]
    labels = np.zeros(N)
    for i in range(len(regions)-1):
        labels[regions[i]:regions[i+1]] = i+1
    n_intervals = len(np.unique(labels))
    
    ## Step 2: Figure out colors for each region
    if (len(colors) == 0):
        N = X.shape[0]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [int(c[1:], 16) for c in colors]
        colors = [((c//65536), ((c%65536)//256), (c%256)) for c in colors]
        colors = colors[1:] + [colors[0]]
        colors = colors*(1+n_intervals//len(colors))
        colors = colors[0:n_intervals]
    elif len(colors) != n_intervals:
        raise ValueError("Must provide same number of colors as intervals")
    
    ## Step 3: Smooth curve and make plot
    t1 = np.linspace(0, 1, X.shape[0])
    t2 = np.linspace(0, 1, fac*X.shape[0])
    spline = CubicSpline(t1, X, axis=0)
    X = spline(t2)
    clabels = (labels[:, None]*np.ones((1, fac))).flatten()

    mol_plot = mlab.plot3d(X[:, 0], X[:, 1], X[:, 2], clabels, colormap='magma', tube_radius=0.5, representation='surface')
    lut = mol_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    interval = 256//n_intervals
    for i in range(n_intervals):
        lut[i*interval:(i+1)*interval, 0:3] = colors[i]
    lut[interval*n_intervals:, 0:3] = colors[-1]
    mol_plot.module_manager.scalar_lut_manager.lut.table = lut

    scene = mlab.gcf().scene
    scene.parallel_projection = True
    scene.background = (1, 1, 1)




class Plotter:
    def __init__(self):
        self.windings = {}
        self.regressions = {}
        self.slopes = {}

    def load(self, windings, regressions, slopes):
        self.windings.update(windings)
        self.regressions.update(regressions)
        self.slopes.update(slopes)

    def plot_regressions(self, save = False, directory = '', progress = True):
        from tqdm import tqdm
        for key in (tqdm(self.regressions, desc = 'Making plots') if (save and progress) else self.regressions):
            plt.clf()
            plot_regression(plt.gca(), self.regressions[key], self.windings[key], self.slopes[key])
        if save:
            plt.savefig(os.path.join(directory, key + '.pdf'))
            plt.close()
        else:
            plt.show()
        
