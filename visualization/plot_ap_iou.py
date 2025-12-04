import numpy as np
import matplotlib.pyplot as plt
with open ('./outtt.txt','r',encoding='utf-8') as file:
    ap = []
    iou = []
    mean_ap = []
    mean_iou = []
    state = 'nothing'
    for line in file.readlines():
        if line[:2] == 'AP':
            state = 'AP'
            ap = []
        elif line[:3] == 'IoU':
            state = 'IoU'
            iou = []
        else:
            if line == '\n':
                if state == 'AP':
                    mean_ap.append(sum(ap)/len(ap))
                elif state == 'IoU':
                    mean_iou.append(sum(iou)/len(iou))
                state = 'nothing'
            if state == 'AP':
                ap.append(float(line))
            elif state == 'IoU':
                iou.append(float(line))

plt.plot(mean_ap, label='AP')
plt.plot(mean_iou, label='IoU')
plt.xlabel('Iter')
plt.ylabel('Value')
plt.title('Mean AP and IoU over Iterations')
plt.legend()
plt.grid()
plt.savefig('mean_ap_iou.png')