r = 20
a,b = eval(input("請輸入點座標 :"))
c = (a**2+b**2)**0.5
if (c <= r):
    print("點座標 %d , %d 在園內部"%(a,b))
else:
    print("點座標 %d , %d 不在園內部"%(a,b))
