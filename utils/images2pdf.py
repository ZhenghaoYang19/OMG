from fpdf import FPDF

def images2pdf(output_file, image_files, page_width=None, page_height=None):
    """Convert a list of images to PDF"""
    if not image_files:
        return
        
    # Create PDF object
    pdf = FPDF()
    
    # Get dimensions of first image if not specified
    if page_width is None or page_height is None:
        import cv2
        img = cv2.imread(image_files[0])
        if img is None:
            raise ValueError(f"Could not read image: {image_files[0]}")
        page_height, page_width = img.shape[:2]
    
    # Convert dimensions to mm (assuming 72 DPI)
    width_mm = page_width * 25.4 / 72
    height_mm = page_height * 25.4 / 72
    
    try:
        # Try newer FPDF version syntax
        pdf.add_page(format=(width_mm, height_mm))
    except TypeError:
        # Fallback for older FPDF versions
        pdf = FPDF(unit='mm', format=[width_mm, height_mm])
        pdf.add_page()
    
    for image in image_files:
        try:
            # Add image to page
            pdf.image(image, 0, 0, width_mm, height_mm)
            # Add new page for next image
            if image != image_files[-1]:  # Don't add page after last image
                try:
                    pdf.add_page(format=(width_mm, height_mm))
                except TypeError:
                    pdf.add_page()
        except Exception as e:
            print(f"Error adding image {image} to PDF: {str(e)}")
            continue
    
    # Save PDF
    try:
        pdf.output(output_file)
    except Exception as e:
        raise Exception(f"Error saving PDF: {str(e)}")
    