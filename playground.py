import numpy as np
import cv2
from collections import OrderedDict

d={0:(5,5),1:(15,2),2:(0,15),3:(5,12),4:(27,5),5:(20,3)}
d = {0:[15,2], 1:[20,3], 2:[5,5], 3:[27,5], 4:[5,12], 5:[0,15] }
print(d.items())
e = []
a = dict(sorted(d.items(), key=lambda item: (item[1][0]+item[1][1])))
b = dict(sorted(d.items(), key=lambda item: (item[1][0]-item[1][1])))
e.extend(sorted(d.items(), key=lambda item: (item[1][0]-item[1][1])))
c = a[list(a.keys())[0]]
d = b[list(b.keys())[-1]]
print(c)
print(d)
print(e)
#a = OrderedDict(sorted(d.items(), key=lambda x: x[0]))