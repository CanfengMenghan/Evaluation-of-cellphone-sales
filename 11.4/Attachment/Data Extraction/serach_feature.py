import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\detail_numerical.xls')
sheet=ExcelFile.sheet_by_name('My Worksheet')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
for i in range (1,2291):
    temp = str(sheet.cell(i,7).value)
    if ('Gravity Response' in temp) :
        worksheet.write(i, 0, label = str(1))
    else:
        worksheet.write(i, 0, label = str(0))
    if ('GPRS' in temp):
        worksheet.write(i, 1, label = str(1))
    else:
        worksheet.write(i, 1, label = str(0))
        '''
    if ('Blue' in temp) or ('blue' in temp):
        worksheet.write(i, 2, label = str(1))
    else:
        worksheet.write(i, 2, label = str(0))
    if ('Rose' in temp) or ('rose' in temp):
        worksheet.write(i, 3, label = str(1))
    else:
        worksheet.write(i, 3, label = str(0))
    if ('Gold' in temp) or ('gold' in temp) or ('champange' in temp) or ('Champange' in temp):
        worksheet.write(i, 4, label = str(1))
    else:
        worksheet.write(i, 4, label = str(0))
    if ('Silver' in temp) or ('silver' in temp):
        worksheet.write(i, 5, label = str(1))
    else:
        worksheet.write(i, 5, label = str(0))
    if ('Grey' in temp) or ('grey' in temp) or ('titanium' in temp) or ('Titanium' in temp):
        worksheet.write(i, 6, label = str(1))
    else:
        worksheet.write(i, 6, label = str(0))
    if ('Pink' in temp) or ('pink' in temp):
        worksheet.write(i, 7, label = str(1))
    else:
        worksheet.write(i, 7, label = str(0))
    if ('Brown' in temp) or ('brown' in temp):
        worksheet.write(i, 8, label = str(1))
    else:
        worksheet.write(i, 8, label = str(0))
    if ('Orange' in temp) or ('orange' in temp):
        worksheet.write(i, 9, label = str(1))
    else:
        worksheet.write(i, 9, label = str(0))
    if ('Yellow' in temp) or ('yellow' in temp):
        worksheet.write(i, 10, label = str(1))
    else:
        worksheet.write(i, 10, label = str(0))
    if ('Red' in temp) or ('red' in temp):
        worksheet.write(i, 11, label = str(1))
    else:
        worksheet.write(i, 11, label = str(0))
    '''
workbook.save('Excel_Workbook.xls')
