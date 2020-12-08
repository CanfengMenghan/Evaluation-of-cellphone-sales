import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\sumai.xlsx')
sheet=ExcelFile.sheet_by_name('速卖')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
row=0
for i in range (1,2291):
    count=0
    for j in range (0,19):
        temp = str(sheet.cell(i,j).value)
        if float(temp)==0.0:
            count=count+1
        #print(j)
    if count==1:
        row=row+1
        for j in range (0,56):
            temp = str(sheet.cell(i,j).value)
            worksheet.write(row, j, label = str(temp))
        worksheet.write(row, 56, label = str(i+1))
    #print(i)
workbook.save('Excel_Workbook.xls')
