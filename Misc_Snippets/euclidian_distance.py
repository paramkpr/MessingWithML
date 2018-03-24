from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

plot_1 = [1, 3]
plot_2 = [2, 5]

distance = sqrt((plot_1[0] - plot_2[0]) ** 2 + (plot_1[1] - plot_2[1]) ** 2)
print(distance)

plt.scatter(plot_1, plot_2)
plt.plot(plot_1, plot_2, label=distance)
plt.legend(loc=4)
plt.show()
