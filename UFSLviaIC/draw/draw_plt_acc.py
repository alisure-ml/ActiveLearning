import os
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools


x_data = ["64", "128", '256', '512', '1024', '2048']

y_data1 = [0.80, 0.73, 0.60, 0.63, 0.55, 0.51][::-1]
y_data2 = [0.82, 0.71, 0.63, 0.62, 0.59, 0.55][::-1]
y_data3 = [0.90, 0.83, 0.70, 0.73, 0.65, 0.61][::-1]
y_data4 = [0.92, 0.81, 0.73, 0.72, 0.69, 0.65][::-1]

plt.figure(figsize=(8, 6))

ln1, = plt.plot(x_data, y_data1, color='red', linewidth=2.0, linestyle='-')
ln2, = plt.plot(x_data, y_data2, color='red', linewidth=2.0, linestyle='-.')
ln3, = plt.plot(x_data, y_data3, color='blue', linewidth=2.0, linestyle='-')
ln4, = plt.plot(x_data, y_data4, color='blue', linewidth=2.0, linestyle='-.')


plt.legend(handles=[ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4, ln1],
           labels=["AAA", "BBB", "CCC", "AAA", "BBB", "CCC",
                   "AAA", "BBB", "CCC"], loc='best', ncol=3, fontsize=16)
plt.grid(linestyle='--')


plt.ylim(0.4, 1.0)
plt.locator_params("y", nbins=10)
plt.xlabel('Dimension', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.tick_params(labelsize=16)
plt.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.98, hspace=0, wspace=0)


plt.savefig(Tools.new_dir(os.path.join("plot", "ic", "demo_acc.pdf")))
plt.show()
