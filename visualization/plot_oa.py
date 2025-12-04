import matplotlib.pyplot as plt
import numpy as np
import math
mx = 10000
y = []
window_size = 100
max_oa = -1e6
with open('../logs/results.out', 'r', encoding='utf-8') as file:
    cnt = 0
    while True:
        line = file.readline()
        if not line:  # 如果读取到文件末尾
            break
        if cnt >= mx:
            break   
        if not line.startswith('Over'):
            continue
        else:
            l = float(line.split(' ')[-1])
            if l < 0.9:
                continue
            if l > max_oa:
                max_oa = l
            y.append(l)
            cnt += 1

y = y[-11:]
plt.plot(range(len(y)), y)
plt.xlabel('Iter')
plt.ylabel('OA')
plt.title('OA Curve')
plt.savefig('oa.png')
print(max_oa)