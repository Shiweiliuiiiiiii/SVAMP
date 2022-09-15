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


dense = [78.64]

results_dir = "/home/sliu/project_space/pruning_cfails/Math/gts/asdiv/"
output_files = os.listdir(os.path.join(results_dir))
all_methods = sorted_nicely(output_files)


results = []
for method in all_methods:
    if 'dense' not in method and 'imp' not in method:
        print(method)
        method_dirs = os.path.join(results_dir, str(method))
        method_sparsity = sorted_nicely(os.listdir(method_dirs))

        for sparsity in method_sparsity:
            sparsity_dir = os.path.join(method_dirs, sparsity,'out', 'CV_results_cv_asdiv-a.json')
            with open(sparsity_dir) as file:
                for line in file:
                    if '5-fold avg acc score' in line:
                        results.append(100*float(line.split()[-1][1:-2]))




# gmp
# omp_after
# omp_before
# omp_rigl
# random_after
# random_before
# random_rigl
# snip
# snip_rigl
robert_gmp = results[:10]
robert_gm_after = results[10:20]
robert_gm_before = results[20:30]
robert_gm_rigl = results[30:40]

robert_random_after = results[40:50]
robert_random_before = results[50:60]
robert_random_rigl = results[60:70]
robert_snip= results[70:80]
robert_snip_rigl= results[80:90]

robert_lth = [80.27, 79.38, 77.32, 71.90, 65.08, 59.57, 56.86, 54.15, 54.81, 52.67]

x_axis = range(10)


roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after,  '-o',   label='One-Shot LRR (After)',color='#77AC30',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, robert_snip,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_random_before,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gm_before,  '-o',   label='OMP (Before)' ,color='cyan',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl,  '--o',   label='OMP+RIGL (Before)',color='cyan',linewidth=linewidth, markersize=markersize )



roberta_large.set_title('GTS on ASDiv-A',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('gts_asdiv.png')
plt.show()