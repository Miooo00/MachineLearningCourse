# 练习2
import math

last = 0
x = 0
while True:
    x = (6 - pow(x, 3) - math.exp(x)/2)/5
    if abs(x - last) < 0.0001:
        break
    last = x
    x += 0.00001
print(x)