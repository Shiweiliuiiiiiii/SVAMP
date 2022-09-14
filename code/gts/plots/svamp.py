import matplotlib.pyplot as plt
import numpy as np
import os, re
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 8
linewidth = 2

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


dense = [41.2]


# robert_gm_before =     [23.1, 26.2, 24.7, 27.0, 25.6, 24.6, 24.7, 24.1, 22.9, 21.9]
robert_snip_rigl =     [27.6, 27.5, 27.6, 26.2, 25.8, 26.0, 26.2, 25.7, 25.4, 24.8]
robert_snip =          [26.9, 26.6, 27.4, 26.2, 26.0, 25.6, 26.4, 25.8, 25.1, 26.2]
robert_lth =           [39.8, 40.1, 39.1, 35.7, 33.7, 25.6, 24.4, 23.6, 22.3, 21.3]


robert_random_before = [23.1, 26.2, 24.7, 27.0, 25.6, 24.6, 24.7, 24.1, 22.9, 21.9]
robert_random_after  = [25.2, 21.4, 25.5, 25.7, 25.6, 25.3, 25.3, 24.2, 23.0, 22.6]
robert_random_rigl =   [28.4, 21.5, 25.3, 27.5, 26.2, 24.7, 25.0, 23.3, 24.0, 22.6]

x_axis = range(10)

# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip,  '-',   label='SNIP',color='blue',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp,  '-',   label='GMP',color='yellow',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl,  '-',   label='SNIP+RIGL',color='blue',linewidth=linewidth, markersize=markersize, marker='^'  )
roberta_large.plot(x_axis, robert_lth,  '-',   label='LTH',color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_before,  '-',   label='OMP Before',color='green',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_after,  '--',   label='OMP After',color='green',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_rigl,  '-',   label='OMP+RIGL',color='green',linewidth=linewidth, markersize=markersize, marker='^' )
roberta_large.plot(x_axis, robert_random_before,  '-',   label='Random Before',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '--',   label='Random After',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl,  '-',   label='Random+RIGL',color='purple',linewidth=linewidth, markersize=markersize, marker='^' )


roberta_large.set_title('GTS on SVAMP',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('gts_SVAMP.png')
plt.show()