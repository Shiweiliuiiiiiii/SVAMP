import matplotlib.pyplot as plt
import numpy as np
import os, re
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 4.5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 15
markersize = 4
linewidth = 1.5

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)



# asdiv 

dense_asdiv = [78.64]

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

robert_gmp_asdiv = results[:10]
robert_gm_after_asdiv = results[10:20]
robert_gm_before_asdiv = results[20:30]
robert_gm_rigl_asdiv = results[30:40]

robert_random_after_asdiv = results[40:50]
robert_random_before_asdiv = results[50:60]
robert_random_rigl_asdiv = results[60:70]
robert_snip_asdiv= results[70:80]
robert_snip_rigl_asdiv= results[80:90]

robert_lth_asdiv = [80.27, 79.38, 77.32, 71.90, 65.08, 59.57, 56.86, 54.15, 54.81, 52.67]



x_axis = range(10)
roberta_large = fig.add_subplot(1,3,1)
roberta_large.plot(x_axis, dense_asdiv*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_asdiv,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_asdiv,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_asdiv,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_asdiv,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_asdiv,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_asdiv,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_asdiv,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, robert_gmp_asdiv,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_asdiv,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_asdiv,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on ASDiv-A',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


# mawps

dense_mawps = [88.49]

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
robert_gmp_mawps = results[:10]
robert_omp_after_mawps = results[10:20]
robert_random_after_mawps = results[20:30]
robert_random_rigl_mawps = results[30:40]
robert_snip_mawps = results[40:50]
robert_snip_rigl_mawps = results[50:60]
robert_random_before_mawps = [81.14, 79.32, 81.51, 82.5, 81.35, 81.30, 80.94, 80.36, 79.58, 80.05]

robert_lth_mawps = [88.28, 88.43, 87.71, 86.98, 84.22, 83.13, 81.93, 81.41, 80.36, 80.73]
robert_omp_before_mawps = [87.97, 85.83, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]
robert_omp_rigl_mawps  = [87.45, 84.16, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]


roberta_large = fig.add_subplot(1,3,2)
roberta_large.plot(x_axis, dense_mawps*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_mawps,  '-o',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_mawps,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_mawps,  '--o',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_omp_after_mawps,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_mawps,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_mawps,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_mawps,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, robert_gmp_asdiv,  '-o', color='magenta',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_before_mawps,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_rigl_mawps,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on MAWPS',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



# svamp

dense_svamp = [41.2]

# after is rerunning due to the issue of loading function
# gmp is running

robert_gm_before_svamp =     [38.5, 29.1, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_gm_rigl_svamp   =     [37.2, 30.5, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_snip_rigl_svamp =     [27.6, 27.5, 27.6, 26.2, 25.8, 26.0, 26.2, 25.7, 25.4, 24.8]
robert_snip_svamp =          [26.9, 26.6, 27.4, 26.2, 26.0, 25.6, 26.4, 25.8, 25.1, 26.2]
robert_lth_svamp =           [39.8, 40.1, 39.1, 35.7, 33.7, 25.6, 24.4, 23.6, 22.3, 21.3]
robert_gmp_svamp =           [41.2, 38.9, 37.5, 33.0, 31.4, 31.6, 28.4, 27.5, 29.1, 30.1]

robert_gm_after_svamp =      [40.9, 39.3, 39.3, 37.5, 29.3, 28.7, 29.9, 28.6, 29.1, 26.0]
robert_random_before_svamp = [23.1, 26.2, 24.7, 27.0, 25.6, 24.6, 24.7, 24.1, 22.9, 21.9]
robert_random_after_svamp  = [34.3, 29.6, 26.6, 26.9, 26.1, 24.5, 25.5, 23.7, 23.2, 20.6]
robert_random_rigl_svamp =   [28.4, 21.5, 25.3, 27.5, 26.2, 24.7, 25.0, 23.3, 24.0, 22.6]

x_axis = range(10)
roberta_large = fig.add_subplot(1,3,3)
roberta_large.plot(x_axis, dense_svamp*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_svamp,  '-o',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_svamp,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_svamp,  '--o',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_svamp,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_svamp,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_svamp,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_svamp,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, robert_gmp_asdiv,  '-o', color='magenta',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_svamp,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_svamp,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on SVAMP',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


# plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05 , bottom=0.2, right=0.95, top=0.90, wspace=0.3, hspace=0.2)

plt.savefig('gts_all.pdf')
plt.show()