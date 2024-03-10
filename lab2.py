import pandas as pd
import matplotlib.pyplot as plt

df = pd.io.parsers.read_csv("breast-cancer-train.dat")

df[1].hist(bins=10, edgecolor="black")
plt.show()
