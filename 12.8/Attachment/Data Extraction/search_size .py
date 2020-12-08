import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\工作簿1.xlsx')
sheet=ExcelFile.sheet_by_name('Sheet1')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
for i in range (1,2291):
    temp = str(sheet.cell(i,0).value)
    if 'x' in temp:
        temp = str.split(temp,'x')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
        worksheet.write(i, 2, label = str(temp[2]))
    elif '*' in temp:
        temp = str.split(temp,'*')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
        worksheet.write(i, 2, label = str(temp[2]))
    elif 'X' in temp:
        temp = str.split(temp,'X')
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
        worksheet.write(i, 2, label = str(temp[2]))
    else:
        worksheet.write(i, 0, label = str(temp[0]))
        worksheet.write(i, 1, label = str(temp[1]))
        worksheet.write(i, 2, label = str(temp[2]))
        temp=[0,0,0]
    temp[0]=float(temp[0])
    temp[1]=float(temp[1])
    temp[2]=float(temp[2])
    judge=temp[0]*temp[1]*temp[2] 
    if judge<36.8633431902425:
        temp[0]=temp[0]*25.4
        temp[1]=temp[1]*25.4
        temp[2]=temp[2]*25.4
    elif judge<4712.4514674042:
        temp[0]=temp[0]*10
        temp[1]=temp[1]*10
        temp[2]=temp[2]*10
    worksheet.write(i, 3, label = str(temp[0]))
    worksheet.write(i, 4, label = str(temp[1]))
    worksheet.write(i, 5, label = str(temp[2]))  
    #worksheet.write(i, 0, label = str(count))
workbook.save('Excel_Workbook.xls')
