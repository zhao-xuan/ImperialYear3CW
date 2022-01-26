import pickle
# from IPython.display import display
from tabulate import tabulate

pickleFile = open("/Users/zhaoxuan/Downloads/hptuning.pkl", 'rb')
df = pickle.load(pickleFile)
f = open("hp.txt", "w")
f.write(tabulate(df, headers = 'keys', tablefmt = 'psql'))
