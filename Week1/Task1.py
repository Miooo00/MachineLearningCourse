# 练习1
n = int(input("几周?\n"))
num = 1
i = 2
while i <= n:
    num += num * 3
    i += 1
    print(num)
