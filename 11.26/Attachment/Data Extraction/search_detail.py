import xlrd
import xlwt
ExcelFile=xlrd.open_workbook(r'C:\Users\tianzhy\Desktop\sumai.xlsx')
sheet=ExcelFile.sheet_by_name('速卖')
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
properties1=['Unlock Phones','Google Play','Battery Type','Battery Capacity','Display Resolution','Operation System','Feature','SIM Card Quantity','Recording Definition','Touch Screen Type','RAM','ROM','color']

properties2=['Size','Display Size']
properties3=['Camera：','Camera Type','Front Camera：']
properties4=['CPU：Octa Core','CPU：Quad Core','CPU：Dual Core']
for i in range (1,2291):
    temp = sheet.cell(i,6).value
    temp = str.split(temp,'<br>')
    length=len(temp)
    for j in range (0,13):
        vari=0
        for k in range (0,length):
            if properties1[j] in temp[k]:
                vari=temp[k]
        worksheet.write(i, j, label = str(vari))
    vari=0
    for k in range (0,length):
        if properties4[0] in temp[k]:
                vari=properties4[0]
        if properties4[1] in temp[k]:
                vari=properties4[1]
        if properties4[2] in temp[k]:
                vari=properties4[2]
    worksheet.write(i, 13, label = str(vari))
    vari=0
    for k in range (0,length):
        if 'Size' in temp[k]:
            vari=vari+1
            if vari < 3:
                worksheet.write(i, (13+vari), label = str(temp[k]))
    vari=0
    for k in range (0,length):
        if 'Camera：' in temp[k]:
            vari=vari+1
            worksheet.write(i, (15+vari), label = str(temp[k]))
            
workbook.save('Excel_Workbook.xls')
