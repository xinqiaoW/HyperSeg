import matplotlib.pyplot as plt
import numpy as np
import math

mx = 100000
y = []
window_size = 100
with open('../logs/train_default.log', 'r', encoding='utf-8') as file:
    cnt = 0
    while True:
        line = file.readline()
        if not line:  # 如果读取到文件末尾
            break
        if cnt >= mx:
            break   
        if not line.startswith('Loss'):
            continue
        else:
            try:
                l = float(line.split(' ')[-2][5:])
            except:
                continue
            y.append(l)
            cnt += 1

y = np.array(y)[-4358:]
window_y = []
for i in range(len(y)):
    if i < (len(y) - window_size):
        window_y.append(np.mean(y[i:i+window_size]))
    else:
        break


plt.plot(range(len(window_y)), window_y)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss1.png')