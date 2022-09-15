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


dense = [78.64]

results_dir = "/home/sliu/project_space/pruning_cfails/Math/gts/mawps/"
output_files = os.listdir(os.path.join(results_dir))
all_methods = sorted_nicely(output_files)


results = []
for method in all_methods:
    if 'dense' not in method and 'imp' not in method and 'omp_rigl' not in method and 'omp_before' not in method:
        print(method)
        method_dirs = os.path.join(results_dir, str(method))
        method_sparsity = sorted_nicely(os.listdir(method_dirs))

        for sparsity in method_sparsity:
            sparsity_dir = os.path.join(method_dirs, sparsity,'out', 'CV_results_cv_mawps.json')
            with open(sparsity_dir) as file:
                for line in file:
                    if '5-fold avg acc score' in line:
                        results.append(100*float(line.split()[-1][1:-2]))


# gmp
# omp_after
# random_before actually is random after
# random_rigl
# snip_before
# snip_rigl
robert_gmp = results[:10]
arobert_omp_after = results[10:20]
robert_random_after = results[20:30]
robert_random_rigl = results[30:40]
robert_snip = results[40:50]
robert_snip_rigl = results[50:60]
robert_random_before = [81.14, 79.32, 81.51, 82.5, 81.35, 81.30, 80.94, 80.36, 79.58, 80.05]

robert_lth = [88.28, 88.43, 87.71, 86.98, 84.22, 83.13, 81.93, 81.41, 80.36, 80.73]
arobert_omp_before = [87.97, 85.83, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]
arobert_omp_rigl   = [87.45, 84.16, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]

x_axis = range(10)

# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip,  '-',   label='SNIP',color='blue',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp,  '-',   label='GMP',color='yellow',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl,  '-',   label='SNIP+RIGL',color='blue',linewidth=linewidth, markersize=markersize, marker='^'  )
roberta_large.plot(x_axis, robert_lth,  '-',   label='LTH',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before,  '-',   label='OMP Before',color='green',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after,  '--',   label='OMP After',color='green',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl,  '-',   label='OMP+RIGL',color='green',linewidth=linewidth, markersize=markersize, marker='^' )
roberta_large.plot(x_axis, robert_random_before,  '-',   label='Random Before',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '--',   label='Random After',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl,  '-',   label='Random+RIGL',color='purple',linewidth=linewidth, markersize=markersize, marker='^' )


roberta_large.set_title('GTS on MAWPS',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('gts_asdiv.png')
plt.show()
