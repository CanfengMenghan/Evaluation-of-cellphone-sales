import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\detail_numerical.xls')
sheet=ExcelFile.sheet_by_name('My Worksheet')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
for i in range (1,2291):
    temp = str(sheet.cell(i,4).value)
    if 'x' in temp:
        temp = str.split(temp,'x')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
    elif '*' in temp:
        temp = str.split(temp,'*')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
    elif 'X' in temp:
        temp = str.split(temp,'X')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
    else:
        worksheet.write(i, 0, label = str(0))
        worksheet.write(i, 1, label = str(0))
        
    #worksheet.write(i, 0, label = str(count))
workbook.save('Excel_Workbook.xls')
