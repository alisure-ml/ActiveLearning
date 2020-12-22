import os
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools


x_data = ["1", "2", '3', '4', '5', '10', '15', '20', "30", "40", "50", '75', '100', '150', '200']

y_data1 = [0.80, 0.73, 0.60, 0.63, 0.55, 0.51, 0.48, 0.45, 0.43, 0.42, 0.31, 0.30, 0.30, 0.30, 0.39][::-1]
y_data2 = [0.82, 0.71, 0.63, 0.62, 0.59, 0.55, 0.44, 0.42, 0.41, 0.40, 0.32, 0.31, 0.30, 0.30, 0.30][::-1]
y_data3 = [0.90, 0.83, 0.70, 0.73, 0.65, 0.61, 0.58, 0.55, 0.53, 0.52, 0.41, 0.31, 0.30, 0.32, 0.31][::-1]
y_data4 = [0.92, 0.81, 0.73, 0.72, 0.69, 0.65, 0.54, 0.52, 0.51, 0.50, 0.49, 0.48, 0.45, 0.33, 0.32][::-1]

plt.figure(figsize=(6, 6))

ln1, = plt.plot(x_data, y_data1, color='red', linewidth=2.0, linestyle='-')
ln2, = plt.plot(x_data, y_data2, color='red', linewidth=2.0, linestyle='-.')
ln3, = plt.plot(x_data, y_data3, color='blue', linewidth=2.0, linestyle='-')
ln4, = plt.plot(x_data, y_data4, color='blue', linewidth=2.0, linestyle='-.')


plt.legend(handles=[ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4, ln1, ln2, ln3, ln4],
           labels=["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB",
                   "AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"], loc='best', ncol=2, fontsize=14)
plt.grid(linestyle='--')


plt.ylim(0.2, 1.0)
plt.locator_params("y", nbins=10)
# plt.xlabel('Shot', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
plt.tick_params(labelsize=14)
plt.subplots_adjust(top=0.96, bottom=0.06, left=0.08, right=0.98, hspace=0, wspace=0)


plt.savefig(Tools.new_dir(os.path.join("plot", "shot", "demo_shot.pdf")))
plt.show()
