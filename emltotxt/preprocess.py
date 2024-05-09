import email
import os
import tempfile
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_attachments(eml_file_path, output_dir):
    with open(eml_file_path, 'rb') as f:
        msg = email.message_from_binary_file(f)
    
    email_id = os.path.splitext(os.path.basename(eml_file_path))[0]  # Extract email ID from filename
    email_output_dir = os.path.join(output_dir, email_id)  # Create directory path for this email
    os.makedirs(email_output_dir, exist_ok=True)  # Create output directory for this email if it doesn't exist
    
    file_count = {}
    
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        filename = part.get_filename()
        if filename:
            data = part.get_payload(decode=True)
            output_path = os.path.join(email_output_dir, filename)
            base, ext = os.path.splitext(filename)
            if base not in file_count:
                file_count[base] = 1
            else:
                file_count[base] += 1
            output_path = os.path.join(email_output_dir, f"{base}_{file_count[base]}{ext}")
            
            with open(output_path, 'wb') as f:
                f.write(data)
            
            # Convert PDF attachments to image files
            if filename.lower().endswith('.pdf'):
                images = convert_pdf_to_images(output_path, output_dir)
                for i, image in enumerate(images):
                    image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_{file_count[base]}_{i+1}.png")
                    image.save(image_path, 'PNG')
                    text = ocr_image(image_path)
                    text_output_path = os.path.splitext(output_path)[0] + f"_{i+1}.txt"
                    with open(text_output_path, 'w') as f:
                        f.write(text)

            # Perform OCR on image attachments
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_path = output_path
                text = ocr_image(image_path)
                text_output_path = os.path.splitext(output_path)[0] + f"_{file_count[base]}.txt"
                with open(text_output_path, 'w') as f:
                    f.write(text)

def convert_pdf_to_images(pdf_path, output_dir):
    images = convert_from_path(pdf_path, dpi=200, output_folder=output_dir, fmt='png')
    return images

# Example usage:
eml_file_path = '/Users/I748970/Downloads/emltotxt/testfiles/testing3.eml'  # Path to the .eml file containing the email
output_dir = '/Users/I748970/Downloads/emltotxt/attachments'    # Output directory to save the extracted attachments

extract_attachments(eml_file_path, output_dir)
