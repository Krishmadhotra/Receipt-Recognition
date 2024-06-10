import os
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from email import message_from_binary_file
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from email.header import decode_header
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.expanduser('~'), 'upload_directory')
OUTPUT_FOLDER = os.path.join(os.path.expanduser('~'), 'output_directory')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

logging.basicConfig(level=logging.INFO)

countduplicates = 0  # Declared as a global variable

def decode_mime_words(s):
    decoded_fragments = decode_header(s)
    return ''.join([str(fragment, encoding or 'utf-8') if isinstance(fragment, bytes) else fragment
                    for fragment, encoding in decoded_fragments])

def extract_text_from_payload(part):
    encodings = ['utf-8', 'latin-1']
    for encoding in encodings:
        try:
            return part.get_payload(decode=True).decode(encoding)
        except UnicodeDecodeError:
            continue
    return None

def count_img_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    return img_tags

def ocr_image(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

def process_pdf(output_path, base, file_count, email_output_dir):
    images = convert_from_path(output_path, dpi=200, output_folder=email_output_dir, fmt='png')
    texts = []
    for i, image in enumerate(images):
        image_path = os.path.join(email_output_dir, f"{base}_{file_count}_{i+1}.png")
        image.save(image_path, 'PNG')
        text = ocr_image(image_path)
        texts.append(text)
    return texts

def extract_attachments(eml_file_path, output_dir):
    with open(eml_file_path, 'rb') as f:
        msg = message_from_binary_file(f)
    
    email_id = os.path.splitext(os.path.basename(eml_file_path))[0]
    email_output_dir = os.path.join(output_dir, email_id)
    os.makedirs(email_output_dir, exist_ok=True)
    
    file_count = {}
    cid_count = {}
    cid_to_filename = {}
    cidlist = []
    img_tags = []

    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            html_content = extract_text_from_payload(part)
            if html_content:
                img_tags = count_img_tags(html_content)
                
                for img in img_tags:
                    cid = img.get('src')[4:-1]
                    cidlist.append(cid)
                    if cid in cid_count:
                        cid_count[cid] += 1
                    else:
                        cid_count[cid] = 1
    
    extracted_texts = []
    filenames = []
    pdf_tasks = []

    with ThreadPoolExecutor() as executor:
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None and part.get('Content-ID') is None:
                continue
            filename = part.get_filename()
            content_id = part.get('Content-ID')
            if filename:
                filename = decode_mime_words(filename)
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
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')) and len(img_tags) < 1:
                    text = ocr_image(output_path)
                    extracted_texts.append(text)
                    filenames.append(f"{base}_{file_count[base]}{ext}")
                    text_output_path = os.path.splitext(output_path)[0] + f"_{file_count[base]}.txt"
                    with open(text_output_path, 'w') as f:
                        f.write(text)
                else:
                    if len(cidlist) > 0:
                        data = part.get_payload(decode=True)
                        for i in cidlist:
                            if i in content_id:
                                cid_to_filename[i] = filename
                                for k in range(cid_count[i]):
                                    text = ocr_image(output_path)
                                    extracted_texts.append(text)
                                    filenames.append(f"{filename}")
                                    text_output_path = os.path.splitext(output_path)[0] + f"_{k+1}.txt"
                                    with open(text_output_path, 'w') as f:
                                        f.write(text)
                                cidlist = list(filter(lambda a: a != i, cidlist))
                            if not cidlist:
                                break

        for task in pdf_tasks:
            texts = task.result()
            extracted_texts.extend(texts)
            for i, text in enumerate(texts):
                filenames.append(f"{base}_{file_count[base]}_{i+1}.png")
                text_output_path = os.path.join(email_output_dir, f"{base}_{file_count[base]}_{i+1}.txt")
                with open(text_output_path, 'w') as f:
                    f.write(text)
    
    return extracted_texts, filenames, cid_to_filename

def get_embeddings(texts, model):
    embeddings = model.encode(texts)
    return embeddings

def cluster_texts_with_dbscan(texts, eps=0.3, min_samples=2):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    
    return labels, embeddings

@app.route('/', methods=['POST'])
def process_eml():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)
            
            extracted_texts, eml_filenames, cid_to_filename = extract_attachments(file_path, app.config['OUTPUT_FOLDER'])
            all_texts = extracted_texts
            all_filenames = eml_filenames

            if not all_texts:
                return jsonify({"error": "No texts were extracted from the .eml file."}), 400
            
            model = SentenceTransformer('all-mpnet-base-v2')
            embeddings = get_embeddings(all_texts, model)
            eps = 0.3
            min_samples = 2
            labels, embeddings = cluster_texts_with_dbscan(all_texts, eps, min_samples)
            
            clusters = {}
            for i, label in enumerate(labels):
                label = str(label)  # Convert label to string to ensure JSON compatibility
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((all_filenames[i], all_texts[i], embeddings[i]))
            
            results = {"clusters": {}}
            for label, files in clusters.items():
                cluster_info = []
                for file in files:
                    cluster_info.append({"filename": file[0], "text": file[1]})
                results["clusters"][label] = cluster_info
            
            duplicates_info = []
            for label, files in clusters.items():
                if len(files) > 1:
                    if label != '-1':
                        for i in range(len(files)):
                            for j in range(i + 1, len(files)):
                                file1, text1, embedding1 = files[i]
                                file2, text2, embedding2 = files[j]
                                semantic_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                                print(f"Semantic similarity between {file1} and {file2}: {semantic_similarity}")
                                if semantic_similarity >= 0.9:
                                    print(f"Appending duplicates: {file1} and {file2}, Similarity: {semantic_similarity}")

                                    duplicates_info.append({
                                        "file1": file1,
                                        "file2": file2,
                                        "similarity": float(semantic_similarity)
                                    })


            results["duplicates"] = duplicates_info

            return jsonify(results), 200

    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    start_time = time.time()
    app.run(debug=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
   
