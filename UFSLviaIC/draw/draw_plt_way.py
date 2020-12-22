import os
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools


x_data = ["2", '3', '4', '5', '10', '15', '20', "30", "40", "50", '60', '70', '80', '90', '100']

y_data1 = [0.70, 0.53, 0.40, 0.33, 0.25, 0.21, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.10, 0.10, 0.09]
y_data2 = [0.72, 0.51, 0.43, 0.32, 0.29, 0.25, 0.14, 0.12, 0.11, 0.10, 0.12, 0.11, 0.10, 0.10, 0.10]
y_data3 = [0.80, 0.63, 0.50, 0.43, 0.35, 0.31, 0.28, 0.25, 0.23, 0.22, 0.21, 0.11, 0.10, 0.12, 0.11]
y_data4 = [0.82, 0.61, 0.53, 0.42, 0.39, 0.35, 0.24, 0.22, 0.21, 0.20, 0.19, 0.28, 0.25, 0.23, 0.22]

plt.figure(figsize=(6, 6))

ln1, = plt.plot(x_data, y_data1, color='red', linewidth=2.0, linestyle='-')
ln2, = plt.plot(x_data, y_data2, color='red', linewidth=2.0, linestyle='-.')
ln3, = plt.plot(x_data, y_data3, color='blue', linewidth=2.0, linestyle='-')
ln4, = plt.plot(x_data, y_data4, color='blue', linewidth=2.0, linestyle='-.')


plt.legend(handles=[ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4],
           labels=["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB",
                   "AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"], loc='best', ncol=2, fontsize=14)
plt.grid(linestyle='--')


plt.ylim(0.0, 1.0)
plt.locator_params("y", nbins=10)
# plt.xlabel('Shot', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
plt.tick_params(labelsize=14)
plt.subplots_adjust(top=0.96, bottom=0.06, left=0.08, right=0.98, hspace=0, wspace=0)


plt.savefig(Tools.new_dir(os.path.join("plot", "way", "demo_way.pdf")))
plt.show()
