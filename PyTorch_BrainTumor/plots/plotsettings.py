# define format for the plots
#import matplotlib as mpl
#serif
def configure(mpl):
    #mpl.rcParams['text.usetex'] = True
    #params = {'text.latex.preamble' : r'\usepackage{amsmath} \usepackage{lmodern} \boldmath'}
    #mpl.rcParams.update(params)

    mpl.rcParams["font.weight"]="bold"
    mpl.rcParams["font.family"]="serif"
    mpl.rcParams['font.size'] = 15
    mpl.rcParams["axes.labelweight"]="bold"
    mpl.rcParams['figure.figsize'] = [6., 4.5]
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 7
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['xtick.minor.width'] = 1.
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.direction'] = "in"
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 5
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['ytick.minor.visible'] = False
    mpl.rcParams['ytick.direction'] = "in"
    mpl.rcParams['legend.fontsize'] = 15
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.frameon'] = False

    # mpl.rcParams['font.weight'] = 'black'
    # mpl.rcParams['axes.labelweight'] = 'black'
    mpl.rcParams['savefig.format'] = "png"
    
