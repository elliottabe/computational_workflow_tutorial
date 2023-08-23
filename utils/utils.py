import json
import yaml
import os
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

##### Plotting settings ######
# sns.set_theme(context='talk', font='Arial', color_codes=True) # palette="dark", style='white',
sns.set_style("white")
sns.set_context("talk")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size':         12,
                     'axes.labelsize':    12,
                     'axes.linewidth':    2,
                     'xtick.major.size':  3,
                     'xtick.major.width': 2,
                     'ytick.major.size':  3,
                     'ytick.major.width': 2,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'font.sans-serif':   "Arial",
                     'font.family':       "sans-serif",
                     'pdf.fonttype':      42,
                     'xtick.labelsize': 10,
                     'ytick.labelsize': 10,
                    })

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def s2b(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters:
    value (str): input value
    
    Returns:
    bool
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmin(memory_available)


########## Checks if path exists, if not then creates directory ##########
def check_path(basepath, path):
    ''' Created by Elliott Abe 
        Parameters:
            basepath: string or Path object for basepath
            path: string for directory or path to check if directory exists, creates if does not exist.

        Returns:
            final_path: returns datatyep Path of combined path of basepath and path
    '''
    from pathlib import Path
    if (type(basepath)==str):
        if path in basepath:
            return basepath
        elif not os.path.exists(os.path.join(basepath, path)):
            os.makedirs(os.path.join(basepath, path))
            print('Added Directory:'+ os.path.join(basepath, path))
            return Path(os.path.join(basepath, path))
        else:
            return Path(os.path.join(basepath, path))
    else:
        if path in basepath.as_posix():
            return basepath
        elif not (basepath / path).exists():
            (basepath / path).mkdir(exist_ok=True,parents=True)
            print('Added Directory:'+ (basepath / path).as_posix())
            return (basepath / path)
        else:
            return (basepath / path)



def add_colorbar(mappable,linewidth=2,location='right',**kwargs):
    ''' modified from https://supy.readthedocs.io/en/2021.3.30/_modules/supy/util/_plot.html'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False,**kwargs)
    cbar.outline.set_linewidth(linewidth)
    plt.sca(last_axes)
    return cbar


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def nan_helper(y):
    """ modified from: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interp_nans(y):
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.
    Parameters:
    x (np.array): array of values
    y (np.array): array of values to shift, must be same length as x
    maxlag (int): number of lags to shift y prior to testing correlation (default 25)
    
    Returns:
    cc_out (np.array): cross correlation
    lags (range): lag vector
    """
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0, len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))

    return cc_out, lags


def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()
    
def h5load(filename):
    store = pd.HDFStore(filename)
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata


def generate_report(Wiki_Path,Wiki_FigPath,config,version):
    from mdutils.mdutils import MdUtils
    from datetime import datetime

    #### getting the current date and time #####
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%m%d%Y")
    dataset_type = config['args']['dataset_type']
    file_dir = Wiki_Path / '{}'.format(dataset_type) 
    file_dir.mkdir(parents=True, exist_ok=True)
    filename = file_dir / '{}_{}_{}.md'.format(current_date,version,dataset_type)
    
    ##### Create Markdown File #####
    mdFile = MdUtils(file_name=filename.as_posix())

    ##### Add Summary #####
    mdFile.new_header(level=1, title='Summary')
    mdFile.new_line("This text will be edited later to include a summary of the setup and results.")

    ##### Add parameters #####
    mdFile.new_header(level=1, title='Parameters')
    mdFile.new_line("<br/> \n <details> \n <summary>Parameters</summary>")
    params_string = "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in config.items()) + "}"
    mdFile.insert_code('params='+ params_string , language='python')
    mdFile.new_line("</details> ")
    mdFile.new_line('')
    ##### Add Results #####
    mdFile.new_header(level=1, title='Plots')

    FigList = sorted(list(Wiki_FigPath.glob('*.png')))
    for n in range(len(FigList)):
        mdFile.new_line('')
        mdFile.new_line(mdFile.new_inline_image(text='{}'.format(FigList[n].name),path=('./../../computing_tutorial/'+(FigList[n]).as_posix())))
        mdFile.new_line('')
        
    mdFile.create_md_file()



