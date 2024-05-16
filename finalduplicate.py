import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from concurrent.futures import ThreadPoolExecutor
from email import message_from_binary_file
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import difflib

# Decoding from .eml
def extract_text_from_payload(part):
    encodings = ['utf-8', 'latin-1']
    for encoding in encodings:
        try:
            return part.get_payload(decode=True).decode(encoding)
        except UnicodeDecodeError:
            continue
    return None

# Finding img tags for inline image counting
def count_img_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    return len(img_tags)

# OCR
def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# PDF to Image and OCR
def process_pdf(output_path, base, file_count, email_output_dir):
    images = convert_from_path(output_path, dpi=200, output_folder=email_output_dir, fmt='png')
    texts = []
    for i, image in enumerate(images):
        image_path = os.path.join(email_output_dir, f"{base}_{file_count}_{i+1}.png")
        image.save(image_path, 'PNG')
        text = ocr_image(image_path)
        texts.append(text)
    return texts

# Extract attachments from .eml file
def extract_attachments(eml_file_path, output_dir):
    with open(eml_file_path, 'rb') as f:
        msg = message_from_binary_file(f)
    
    email_id = os.path.splitext(os.path.basename(eml_file_path))[0]
    email_output_dir = os.path.join(output_dir, email_id)
    os.makedirs(email_output_dir, exist_ok=True)
    
    file_count = {}
    inline_image_count = 0
    
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            html_content = extract_text_from_payload(part)
            if html_content:
                inline_image_count = count_img_tags(html_content)
    
    extracted_texts = []
    filenames = []
    pdf_tasks = []

    with ThreadPoolExecutor() as executor:
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
                
                if filename.lower().endswith('.pdf'):
                    pdf_tasks.append(executor.submit(process_pdf, output_path, base, file_count[base], email_output_dir))
                elif (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')) and (inline_image_count < 2)):
                    text = ocr_image(output_path)
                    extracted_texts.append(text)
                    filenames.append(f"{base}_{file_count[base]}{ext}")
                    text_output_path = os.path.splitext(output_path)[0] + f"_{file_count[base]}.txt"
                    with open(text_output_path, 'w') as f:
                        f.write(text)
                elif part.get_content_disposition() and inline_image_count > 1:
                    for k in range(inline_image_count):
                        image_data = part.get_payload(decode=True)
                        filename = part.get_filename()
                        output_path = os.path.join(email_output_dir, filename)
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        text = ocr_image(output_path)
                        extracted_texts.append(text)
                        filenames.append(f"{base}_{k}.txt")
                        text_output_path = os.path.splitext(output_path)[0] + f"_{k}.txt"
                        with open(text_output_path, 'w') as f:
                            f.write(text)

        for task in pdf_tasks:
            texts = task.result()
            extracted_texts.extend(texts)
            for i, text in enumerate(texts):
                filenames.append(f"{base}_{file_count[base]}_{i+1}.png")
                text_output_path = os.path.join(email_output_dir, f"{base}_{file_count[base]}_{i+1}.txt")
                with open(text_output_path, 'w') as f:
                    f.write(text)
    
    return extracted_texts, filenames

# Function to compute embeddings
def get_embeddings(texts, model):
    embeddings = model.encode(texts)
    return embeddings

# Function to print differences between two texts
def print_differences(text1, text2, file1, file2):
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    differences = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
    if differences:
        print(f"Differences between {file1} and {file2}:")
        print("\n".join(differences))
        print("\n *** NO DUPLICATES ***")
    else:
        print("\n !!! DUPLICATES PRESENT !!!")

def main():
    start_time = time.time()
    # Load Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    eml_file_path = '/Users/I748970/Downloads/emltotxt/testfiles/testing3.eml'
    output_dir = '/Users/I748970/Downloads/emltotxt/attachments'
    
    # Extract texts and filenames
    extracted_texts, filenames = extract_attachments(eml_file_path, output_dir)
    
    # Debug extraction
    if not extracted_texts:
        print("No texts were extracted.")
        return
    
    # Get embeddings
    embeddings = get_embeddings(extracted_texts, model)
    
    # DBSCAN clustering
    dbscan = DBSCAN(metric='cosine', eps=0.3, min_samples=2)
    labels = dbscan.fit_predict(embeddings)
    
    # Print differences for similar files
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((filenames[i], extracted_texts[i], embeddings[i]))
    
    for label, files in clusters.items():
        if len(files) > 1:
            print(f"\nDifferences in Cluster {label}:")
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    file1, text1, embedding1 = files[i]
                    file2, text2, embedding2 = files[j]
                    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                    if similarity > 0.8:
                            print(f"\nSimilarity Score between {file1} and {file2}: {similarity}")
                            print_differences(text1, text2, file1, file2)
                    else:
                        print(f"\nSimilarity Score between {file1} and {file2}: {similarity}")
                        print("\n *** NO DUPLICATES ***")

    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution time:", execution_time)

if __name__ == "__main__":
    main()
