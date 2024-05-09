import email
import os
import tempfile
import pytesseract
import fitz  # Import the PyMuPDF library
from PIL import Image

def extract_attachments(eml_file_path, output_dir):
    with open(eml_file_path, 'rb') as f:
        msg = email.message_from_binary_file(f)
    
    email_id = os.path.splitext(os.path.basename(eml_file_path))[0]  # Extract email ID from filename
    email_output_dir = os.path.join(output_dir, email_id)  # Create directory path for this email
    os.makedirs(email_output_dir, exist_ok=True)  # Create output directory for this email if it doesn't exist
    
    attachments = {}
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        filename = part.get_filename()
        if filename:
            if filename in attachments:
                attachments[filename].append(part.get_payload(decode=True))
            else:
                attachments[filename] = [part.get_payload(decode=True)]
    
    for filename, data_list in attachments.items():
        for i, data in enumerate(data_list):
            output_path = os.path.join(email_output_dir, f"{os.path.splitext(filename)[0]}_{i+1}{os.path.splitext(filename)[1]}")
            with open(output_path, 'wb') as f:
                f.write(data)
        
            # Perform OCR on image attachments
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_path = output_path
                text = ocr_image(image_path)
                text_output_path = os.path.splitext(output_path)[0] + '.txt'
                with open(text_output_path, 'w') as f:
                    f.write(text)
            elif filename.lower().endswith('.pdf'):
                pdf_path = output_path
                text = ocr_pdf(pdf_path)
                text_output_path = os.path.splitext(output_path)[0] + '.txt'
                with open(text_output_path, 'w') as f:
                    f.write(text)


def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def ocr_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            # Get the page
            page = doc.load_page(page_num)
            # Convert the page to an image
            image = page.get_pixmap()
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
                temp_image.write(image.tobytes())
                temp_image_path = temp_image.name
            # Perform OCR on the image
            page_text = ocr_image(temp_image_path)
            text += page_text + "\n"  # Add the text from the page to the overall text
            # Delete the temporary image file
            os.unlink(temp_image_path)
        
    return text


# Example usage:
eml_file_path = '/Users/I749114/Downloads/Emails/63cfbc43-ed85-4763-b0b5-1727006e6c7a-1715224659329.eml'  # Path to the .eml file containing the email
output_dir = '/Users/I749114/Downloads/Attachments'    # Output directory to save the extracted attachments

extract_attachments(eml_file_path, output_dir)
