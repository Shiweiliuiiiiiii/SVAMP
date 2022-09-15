import matplotlib.pyplot as plt
import numpy as np
import os, re
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 4
linewidth = 1.5

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


dense = [41.2]

# after is rerunning due to the issue of loading function
# gmp is running

robert_gm_before =     [38.5, 29.1, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_gm_rigl   =     [37.2, 30.5, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_snip_rigl =     [27.6, 27.5, 27.6, 26.2, 25.8, 26.0, 26.2, 25.7, 25.4, 24.8]
robert_snip =          [26.9, 26.6, 27.4, 26.2, 26.0, 25.6, 26.4, 25.8, 25.1, 26.2]
robert_lth =           [39.8, 40.1, 39.1, 35.7, 33.7, 25.6, 24.4, 23.6, 22.3, 21.3]
robert_gmp =           [41.2, 38.9, 37.5, 33.0, 31.4, 31.6, 28.4, 27.5, 29.1, 30.1]

robert_gm_after =      [40.9, 39.3, 39.3, 37.5, 29.3, 28.7, 29.9, 28.6, 29.1, 26.0]
robert_random_before = [23.1, 26.2, 24.7, 27.0, 25.6, 24.6, 24.7, 24.1, 22.9, 21.9]
robert_random_after  = [34.3, 29.6, 26.6, 26.9, 26.1, 24.5, 25.5, 23.7, 23.2, 20.6]
robert_random_rigl =   [28.4, 21.5, 25.3, 27.5, 26.2, 24.7, 25.0, 23.3, 24.0, 22.6]

x_axis = range(10)

# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after,  '-o',   label='One-Shot LRR (After)',color='#77AC30',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, robert_snip,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_random_before,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gm_before,  '-o',   label='OMP (Before)' ,color='cyan',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl,  '--o',   label='OMP+RIGL (Before)',color='cyan',linewidth=linewidth, markersize=markersize )



roberta_large.set_title('GTS on SVAMP',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('gts_SVAMP.png')
plt.show()