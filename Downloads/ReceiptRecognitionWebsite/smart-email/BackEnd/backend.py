import os
import warnings
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from email import message_from_binary_file
from PIL import Image
import pytesseract
import difflib
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from flask_cors import CORS
import numpy as np 

def convert_float32(obj):
    if isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
warnings.filterwarnings("ignore", category=FutureWarning)

# Helper functions
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
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
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

def cluster_texts_with_dbscan(texts, eps=0.1, min_samples=2):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)

    return labels, embeddings

def compute_similarity_scores(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def compute_detailed_similarity(text1, text2):
    sequence_matcher = difflib.SequenceMatcher(None, text1, text2)
    return sequence_matcher.ratio()

@app.route('/', methods=['POST'])
def process_eml():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.eml'):
        return jsonify({'error': 'Invalid file type. Only .eml files are allowed.'}), 400

    output_dir = 'uploads'
    os.makedirs(output_dir, exist_ok=True)
    eml_file_path = os.path.join(output_dir, file.filename)
    file.save(eml_file_path)

    extracted_texts, filenames, cid_to_filename = extract_attachments(eml_file_path, output_dir)
    if not extracted_texts:
        return jsonify({'error': 'No texts were extracted.'}), 400

    embeddings = get_embeddings(extracted_texts, SentenceTransformer('all-mpnet-base-v2'))
    eps = 0.3
    min_samples = 2
    labels, embeddings = cluster_texts_with_dbscan(extracted_texts, eps, min_samples)
    similarity_matrix = compute_similarity_scores(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((filenames[i], extracted_texts[i], embeddings[i]))

    results = []
    for label, files in clusters.items():
        cluster_info = {'cluster': int(label), 'files': [file[0] for file in files]}
        results.append(cluster_info)

    similarities = []
    similar_files = []
    dissimilar_files = []

    semantic_similarity = None  # Define here
    for label, files in clusters.items():
        if len(files) > 1:
            if label != -1:
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        file1, text1, embedding1 = files[i]
                        file2, text2, embedding2 = files[j]
                        semantic_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                        detailed_similarity = compute_detailed_similarity(text1, text2)
                        similarity_info = {
                            'file1': file1,
                            'file2': file2,
                            'semantic_similarity': float(semantic_similarity),
                            'detailed_similarity': float(detailed_similarity),
                            'are_similar': semantic_similarity > 0.8 and detailed_similarity < 1.0
                        }
                        similarities.append(similarity_info)
                        if similarity_info['are_similar']:
                            dissimilar_files.append({
                                'file1': file1,
                                'file2': file2,
                                # 'semantic_similarity': convert_float32(semantic_similarity),
                                # 'detailed_similarity': convert_float32(detailed_similarity)
                            })
                        else:
                            similar_files.append({
                                'file1': file1,
                                'file2': file2,
                                # 'semantic_similarity': convert_float32(semantic_similarity),
                                # 'detailed_similarity': convert_float32(detailed_similarity)
                            })

    if not similar_files and not dissimilar_files:
      return jsonify({'message': 'Distinct files',
                      "semantic_similarity":convert_float32(semantic_similarity)
                      
                    })
    
    return jsonify({
    'similar_files': similar_files,
    'dissimilar_files': dissimilar_files,
    'semantic_similarity': float(semantic_similarity)  # Convert to float
})

if __name__ == "__main__":
  app.run(debug=True)