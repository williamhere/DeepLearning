print("溫度轉換選擇")
option = eval(input("1:華氏溫度轉成攝氏溫度 \n2:攝氏溫度轉成華氏溫度\n="))
#op = int(option)
if(option == 1):
    F0 = int(input("請輸入華氏溫度:"))
    C0 = (F0-32)*5/9
    print("華氏溫度%0.1F等於攝氏溫度%0.1F"%(F0,C0))
elif(option == 2):
    C1 = int(input("請輸入攝氏溫度:"))
    F1 = (C1*9/5)+32
    print("攝氏溫度%0.1F等於華氏溫度%0.1F"%(C1,F1))
else:
    print("輸入錯誤")
