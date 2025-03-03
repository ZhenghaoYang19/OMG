from fpdf import FPDF

def images2pdf(outpath, images, w = 1920, h = 1080):
    pdf = FPDF()
    pdf.compress = False
    size=(h, w)
    for image in images:
        pdf.add_page(orientation = 'L', format=size, same=False)
        pdf.image(name = image, x = 0, y = 0, w = w, h = h, type = 'JPG')

    pdf.output(outpath, "F")
    