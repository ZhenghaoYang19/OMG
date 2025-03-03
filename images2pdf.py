import os
from fpdf import FPDF

def images2pdf(outpath, images, w = 1920, h = 1080):
    pdf = FPDF()
    pdf.compress = False
    titleH = 60
    size=(h + titleH, w)
    for image in images:
        pdf.add_page(orientation = 'L', format=size, same=False)
        pdf.set_font('helvetica', size = titleH)
        pdf.cell(w = 400, h = titleH, txt = os.path.basename(image), border = 1, align = 'C')
        pdf.image(name = image, x = 0, y = titleH + 10, w = w, h = h, type = 'JPG')

    pdf.output(outpath, "F")
    