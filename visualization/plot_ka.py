import matplotlib.pyplot as plt
import numpy as np
import math
mx = 10000
y = []
window_size = 100
max_ka = -1e6
with open('../logs/results.out', 'r', encoding='utf-8') as file:
    cnt = 0
    while True:
        line = file.readline()
        if not line:  # 如果读取到文件末尾
            break
        if cnt >= mx:
            break   
        if not line.startswith('Kapp'):
            continue
        else:
            l = float(line.split(' ')[-1])
            if l < 0.8:
                continue
            if l > max_ka:
                max_ka = l
            y.append(l)
            cnt += 1


plt.plot(range(len(y)), y)
plt.xlabel('Iter')
plt.ylabel('KA')
plt.title('KA Curve')
plt.savefig('ka.png')
print(max_ka)