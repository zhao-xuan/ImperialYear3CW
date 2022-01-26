import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots()

# Compiler optimisation level
def compiler_op():
    opt_level = ["-O0", "-O1", "-O2", "-O3"]
    avg_runtime = [2.3335, 0.5991, 0.4429, 0.4261]
    fp_perf = [0.0851, 0.3314, 0.4482, 0.4659]

    ax1.plot(opt_level, avg_runtime, marker=".", linestyle="-", color="yellowgreen")
    ax1.set(xlabel='gcc Optimization Level',
        title='Avg. Execution Time of sptMTTKRP function and \n Floating Point Performance vs. Compiler Optimization Level')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution Time (second)")

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(opt_level, fp_perf, marker=".", linestyle="-", color="dodgerblue", label="Floating Point Performance")
    ax2.plot(np.nan, marker=".", linestyle="-", color="yellowgreen", label="Execution Time")
    ax2.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="center right")

def openmp():
    nthreads = [1, 2, 3, 4, 8, 12, 16, 32]
    x = [1,2,3,4,5,6,7,8]
    avg_runtime = [2.6060, 1.5177, 1.0376, 0.9970, 0.9770, 0.9409, 0.9229, 0.8981]
    fp_perf = [0.0762, 0.1308, 0.1913, 0.1991, 0.2031, 0.2110, 0.2151, 0.2210]

    ax1.plot(x, avg_runtime, marker=".", linestyle="-", color="yellowgreen")
    ax1.set(xlabel='Number of Threads using OpenMP',
        title='Avg. Execution Time of sptMTTKRP function and \n Floating Point Performance vs. Thread Count')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution Time (second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nthreads)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(x, fp_perf, marker=".", linestyle="-", color="dodgerblue", label="Floating Point Performance")
    ax2.plot(np.nan, marker=".", linestyle="-", color="yellowgreen", label="Execution Time")
    ax2.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="center right")

def overclocking():
    clock_rate = [1500, 1800, 2000, 2147]
    x = [1,2,3,4]
    avg_runtime = [0.474973980, 0.394572117, 0.362860928, 0.340716116]
    fp_perf = [0.42, 0.5098, 0.5494, 0.58]

    ax1.plot(x, avg_runtime, marker=".", linestyle="-", color="yellowgreen")
    ax1.set(xlabel='Overclocking Rate (MHz)',
        title='Avg. Execution Time of sptMTTKRP function and \n Floating Point Performance versus Clock Rate')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution time (second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clock_rate)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(x, fp_perf, marker=".", linestyle="-", color="dodgerblue", label="Floating Point Performance")
    ax2.plot(np.nan, marker=".", linestyle="-", color="yellowgreen", label="Execution Time")
    ax2.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="lower center")

def opt_steps():
    avg_runtime = [0.4261, 0.2521335, 0.24624, 0.205572, 0.17456]
    fp_perf = [0.4261, 0.78726, 0.805976, 0.965476, 1.136972]
    labels = ["Baseline", "SIMD", "Loop Unrolling", "Loop Fusion", "Array Contraction"]
    x = [1, 2, 3, 4, 5]

    ax1.plot(x, avg_runtime, marker=".", linestyle="-", color="yellowgreen")
    ax1.set(xlabel='Optimization Steps',
        title='Avg. Execution Time of sptMTTKRP function and \n Floating Point Performance versus Optimization Steps')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution time (second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(x, fp_perf, marker=".", linestyle="-", color="dodgerblue", label="Floating Point Performance")
    ax2.plot(np.nan, marker=".", linestyle="-", color="yellowgreen", label="Execution Time")
    ax2.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="lower center")

def opt_overclocking():
    clock_rate = [1500, 1800, 2000, 2147]
    x = [1,2,3,4]
    avg_runtime = [0.204644, 0.17514, 0.159888, 0.15091]
    fp_perf = [0.969958, 1.133332, 1.24144, 1.3153558]

    ax1.plot(x, avg_runtime, marker=".", linestyle="-", color="yellowgreen")
    ax1.set(xlabel='Overclocking Rate (MHz)',
        title='Avg. Execution Time of sptMTTKRP function and \n Floating Point Performance versus Clock Rate')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution time (second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clock_rate)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(x, fp_perf, marker=".", linestyle="-", color="dodgerblue", label="Floating Point Performance")
    ax2.plot(np.nan, marker=".", linestyle="-", color="yellowgreen", label="Execution Time")
    ax2.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax2.tick_params(axis='y')
    ax2.legend(loc="lower center")

def temp():
    opt_temp = [47.407625, 48.3381428571, 51.4456666667, 52.095]
    baseline_temp = [49.0756, 50.212, 50.086125, 52.582]

    x = [1,2,3,4]
    clock_rate = [1500, 1800, 2000, 2147]

    ax1.plot(x, baseline_temp, marker=".", linestyle="-", color="yellowgreen", label="Basline")
    ax1.plot(x, opt_temp, marker=".", linestyle="-", color="dodgerblue", label="Optimised" )
    ax1.set(xlabel='Overclocking Rate (MHz)',
        title='Avg. Temperature of Baseline & Optimised Versions vs. Clock Rate')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Average Temperature (℃)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clock_rate)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()
    ax1.legend(loc="lower center")

def opt_openmp_overclock_runtime():
    x = [1,2,3,4,5,6,7,8]
    nthreads = [1, 2, 3, 4, 8, 12, 16, 32]
    avg_runtime_2147 = [0.149946548, 0.078828910, 0.054512841, 0.043046347, 0.043410743, 0.048812212, 0.048920442, 0.046732239]
    avg_runtime_2000 = [0.158035494, 0.084326011, 0.058306789, 0.045621826, 0.046146998, 0.047614714, 0.049960438, 0.050052573]
    avg_runtime_1800 = [0.173978801, 0.091285593, 0.062203248, 0.048591221, 0.049341288, 0.053652715, 0.054837290, 0.051445926]
    avg_runtime_1500 = [0.201902581, 0.105959934, 0.072359188, 0.055998565, 0.066070479, 0.060295299, 0.060433282, 0.061377637]

    ax1.plot(x, avg_runtime_2147, marker=".", linestyle="-", color="darkorchid", label="2147 MHz")
    ax1.plot(x, avg_runtime_2000, marker=".", linestyle="-", color="yellowgreen", label="2000 MHz")
    ax1.plot(x, avg_runtime_1800, marker=".", linestyle="-", color="mediumturquoise", label="1800 MHz")
    ax1.plot(x, avg_runtime_1500, marker=".", linestyle="-", color="dodgerblue", label="1500 MHz")
    
    ax1.set(xlabel='Number of Threads',
        title='Avg. Execution Time of sptMTTKRP function and Floating Point \n Performance versus No. of Threads under Different Clock Rate')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Execution time (second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nthreads)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()
    ax1.legend(loc="center right")

def opt_openmp_overclock_flops():
    x = [1,2,3,4,5,6,7,8]
    nthreads = [1, 2, 3, 4, 8, 12, 16, 32]
    avg_flops_2147 = [1.3238249157, 2.5181494390, 3.6413984734, 4.6113779643, 4.5726693755, 4.0666662345, 4.0576692907, 4.1645341879]
    avg_flops_2000 = [1.2560657766, 2.3539946174, 3.4044573320, 4.3510528873, 4.3015360609, 4.1689418597, 3.9732032456, 3.9658895616]
    avg_flops_1800 = [1.1409607082, 2.1745268734, 3.1911995130, 4.0851612757, 4.0230602491, 3.6997750578, 3.6198538754, 3.8584780305]
    avg_flops_1500 = [0.9831621499, 1.8733776863, 2.7433002302, 3.5447868098, 3.0044125456, 3.2921799976, 3.2846631544, 3.2341253983]

    ax1.plot(x, avg_flops_2147, marker=".", linestyle="-", color="darkorchid", label="2147 MHz")
    ax1.plot(x, avg_flops_2000, marker=".", linestyle="-", color="yellowgreen", label="2000 MHz")
    ax1.plot(x, avg_flops_1800, marker=".", linestyle="-", color="mediumturquoise", label="1800 MHz")
    ax1.plot(x, avg_flops_1500, marker=".", linestyle="-", color="dodgerblue", label="1500 MHz")

    ax1.set(xlabel='Number of Threads',
        title='Floating Point Performance and Floating Point Performance \n versus No. of Threads under Different Clock Rate')
    ax1.tick_params(axis='y')
    ax1.set_ylabel("Floating Point Performance (GFLOP/s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nthreads)

    ax1.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax1.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax1.minorticks_on()
    ax1.legend(loc="lower center")

opt_openmp_overclock_flops()
fig.tight_layout()
plt.show()