import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\sumai.xlsx')
sheet=ExcelFile.sheet_by_name('速卖')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
for i in range (1,2291):
    temp = sheet.cell(i,5).value
    temp = str.split(temp)
    length=len(temp)
    count=0
    for j in range (0,length):
        if temp[j]=='Xiaomi'or temp[j]=='xiaomi'or temp[j]=='XIAOMI':
            count=1
        elif temp[j]=='Huawei'or temp[j]=='HUAWEI'or temp[j]=='huawei':
            count=2
        elif temp[j]=='MEIZU'or temp[j]=='meizu'or temp[j]=='Meizu':
            count=3
        elif temp[j]=='LENOVO'or temp[j]=='Lenovo'or temp[j]=='lenovo':
            count=4
        elif temp[j]=='IPHONE'or temp[j]=='iphone'or temp[j]=='iPhone':
            count=5
        elif temp[j]=='OPPO'or temp[j]=='Oppo'or temp[j]=='oppo':
            count=6
        elif temp[j]=='Vivo'or temp[j]=='vivo'or temp[j]=='VIVO':
            count=7
        elif temp[j]=='Nubia'or temp[j]=='NUBIA'or temp[j]=='nubia':
            count=8
        elif temp[j]=='samsung'or temp[j]=='Samsung'or temp[j]=='SAMSUNG':
            count=9
        elif temp[j]=='ZTE'or temp[j]=='zte':
            count=10
        elif temp[j]=='HOMTOM'or temp[j]=='homtom'or temp[j]=='Homtom':
            count=11
        elif temp[j]=='DOOGEE'or temp[j]=='Doogee'or temp[j]=='doogee':
            count=12
        elif temp[j]=='Letv'or temp[j]=='LeTv'or temp[j]=='letv' or temp[j]=='LETV':
            count=13
        elif temp[j]=='Blackview'or temp[j]=='BLACKVIEW'or temp[j]=='blackview':
            count=14
        elif temp[j]=='NOKIA'or temp[j]=='Nokia'or temp[j]=='nokia':
            count=15
    worksheet.write(i, 0, label = str(count))
workbook.save('Excel_Workbook.xls')
