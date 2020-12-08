import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\sumai.xlsx')
sheet=ExcelFile.sheet_by_name('速卖')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
for i in range (1,2291):
    vari=0
    temp = sheet.cell(i,6).value
    if 'Dual Camera' in temp:
        vari=1
    if 'Dual camera' in temp:
        vari=1
    if 'Dual Front Camera' in temp:
        vari=1
    if 'Dual Back Camera' in temp:
        vari=1
    if 'Dual front Camera' in temp:
        vari=1
    if 'Dual Rear Camera' in temp:
        vari=1
    if 'Dual rear Camera' in temp:
        vari=1
    if 'Dual back Camera' in temp:
        vari=1
    worksheet.write(i, 0, label = vari)
    vari=0
    if 'Front Camera' in temp:
        vari=1
    if 'front Camera' in temp:
        vari=1
    worksheet.write(i, 1, label = vari)  
workbook.save('Excel_Workbook.xls')
