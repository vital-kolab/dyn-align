import matplotlib as mpl

import matplotlib.pyplot as plt


def journal_figure_pdf(do_save=False, filename='figure.eps', dpi=300, size_inches=(2.16, 2.16), linewidth=2, fontsize_x=12, fontsize_y=12, family='Arial'):
    """
    Adjusts the current matplotlib figure to make it look publication-worthy.
    
    Parameters:
    - do_save: bool, whether to save the figure to an EPS file.
    - filename: str, the name of the file to save the figure as.
    - dpi: int, the resolution of the figure in dots per inch.
    - size_inches: tuple, the size of the figure in inches.
    - linewidth: float, the line width for the plot elements.
    """
    ax = plt.gca()  # Get the current axes
    
    # Adjust tick direction and length
    ax.tick_params(direction='out', length=10, width=linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    
    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=2)
    ax.set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
    # Set font size and type
    #plt.xticks(fontsize=12, fontname='Times New Roman')
    #plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.xticks(fontsize=fontsize_x, family=family)
    plt.yticks(fontsize=fontsize_y, family=family)

    ax.set_xlabel(ax.get_xlabel(), family=family)
    ax.set_ylabel(ax.get_ylabel(), family=family)

    ax.set_title(ax.get_title(), family=family)

    if not ax.get_legend_handles_labels() == ([], []) :
        L = ax.legend()
        plt.setp(L.texts, family=family)

    im = ax.images        
    # Assume colorbar was plotted last one plotted last
    if len(im) > 0:      
        # Assume colorbar was plotted last one plotted last
        cb = im[-1].colorbar  
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_fontproperties(family)
            l.set_fontsize(fontsize_x)
    
    if do_save:
        # Save the figure
        #plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='eps', linewidth=linewidth)
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='pdf')