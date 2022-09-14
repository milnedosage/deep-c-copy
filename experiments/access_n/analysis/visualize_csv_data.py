import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame
filename = input("Enter name of csv file: ")
try:
    df = pd.read_csv(filename)
except FileNotFoundError as e:
    print(e)
    exit()

# Filter only large array sizes
is_large_n = df["n"] > 1e10
df = df[is_large_n]

# Get both columns 
array_size = df["n"]
duration_ns = df["duration_ns"]

# Simple statistics
n_max = array_size.max()
n_min = array_size.min()
n_mean = array_size.mean()
n_std = array_size.std()

t_max = duration_ns.max()
t_min = duration_ns.min()
t_mean = duration_ns.mean()
t_std = duration_ns.std()

print("\nStatistics for filtered data (i.e. array size > 1e10)")
print(f"Array size n:\nMax - {n_max:.2f}\tMin - {n_min:.2f}\tMean - {n_mean:.2f}\tStd - {n_std:.2f}")
print(f"Time (ns):\nMax - {t_max:.2f}\tMin - {t_min:.2f}\tMean - {t_mean:.2f}\tStd - {t_std:.2f}\n")

# Get top array sizes with fastest and slowest performance
size = int(input("Enter how many different sizes you would like to see: "))
is_fastest_time = duration_ns.nsmallest(size)
fastest_times = df.loc[is_fastest_time.keys()]
fastest_array_sizes = fastest_times["n"]

is_slowest_time = duration_ns.nlargest(size)
slowest_times = df.loc[is_slowest_time.keys()]
slowest_array_sizes = slowest_times["n"]

fastest_array_sizes.index = range(size)
slowest_array_sizes.index = range(size)
combined = pd.concat([fastest_array_sizes, slowest_array_sizes], axis=1)
combined.columns = ["Fastest", "Slowest"]
print(f"Array sizes (descending order):\n{combined}")

# Plotting scatter plot of array size vs time
plt.title("Time as a function of array size")
plt.xlabel("array size n")
plt.ylabel("time (ns)")
plt.scatter(array_size, duration_ns)
plt.savefig("n vs time.pdf")