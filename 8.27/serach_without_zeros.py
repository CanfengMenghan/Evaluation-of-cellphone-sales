import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\detail_final_select.xlsx')
ExcelFile2=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\detail_final.xlsx')
sheet=ExcelFile.sheet_by_name('My Worksheet')
sheet2=ExcelFile2.sheet_by_name('My Worksheet')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
row=0
for i in range (1,2291):
    check=1
    for j in range (0,20):
        temp = str(sheet.cell(i,j).value)
        if float(temp)==0.0:
            check=0
    if check==1:
        row=row+1
        for j in range (0,34):
            temp = str(sheet2.cell(i,j).value)
            worksheet.write(row, j, label = str(temp))
        worksheet.write(row, 34, label = str(i+1))
    #print(i)
workbook.save('Excel_Workbook.xls')
