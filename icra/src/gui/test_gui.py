import matplotlib.pyplot as plt
from itertools import count
from realtime_plotter import RealtimePlotter
import matplotlib.gridspec as gridspec

# Initialize Figure
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)

def plotit():
    rplt = RealtimePlotter(fig, gs)
    for i in count(1):
        rplt.update(i)


plotit()
