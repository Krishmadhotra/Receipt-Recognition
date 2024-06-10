import os
import time
import warnings
import pytesseract
import glob
from flask import Flask, request, jsonify
from email import message_from_binary_file
from email.header import decode_header
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from datasketch import MinHash, MinHashLSH
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'eml'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

# Utility functions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def process_pdf(output_path, filename, file_count, image_output_dir, text_output_dir):
    images = convert_from_path(output_path, dpi=200, fmt='png')
    texts = []
    for i, image in enumerate(images):
        image_path = os.path.join(image_output_dir, f"{filename}_{file_count}_{i+1}.png")
        image.save(image_path, 'PNG')
        text = ocr_image(image_path)
        texts.append(text)
        text_output_path = os.path.join(text_output_dir, f"{filename}_{file_count}_{i+1}.txt")
        with open(text_output_path, 'w') as f:
            f.write(text)
    return texts

def extract_attachments(eml_file_path, output_dir):
    with open(eml_file_path, 'rb') as f:
        msg = message_from_binary_file(f)
    
    email_id = os.path.splitext(os.path.basename(eml_file_path))[0]
    email_output_dir = os.path.join(output_dir, email_id)
    text_output_dir = os.path.join(email_output_dir, 'texts')
    image_output_dir = os.path.join(email_output_dir, 'images')
    
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
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
                output_path = os.path.join(image_output_dir, f"{base}_{file_count[base]}{ext}")
                
                with open(output_path, 'wb') as f:
                    f.write(data)
                
                if filename.lower().endswith('.pdf'):
                    pdf_tasks.append(executor.submit(process_pdf, output_path, filename, file_count[base], image_output_dir, text_output_dir))
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        if len(cidlist) > 0:
                            data = part.get_payload(decode=True)
                            for i in cidlist:
                                if i in content_id:
                                    cid_to_filename[i] = filename
                                    for k in range(cid_count[i]):
                                        text = ocr_image(output_path)
                                        extracted_texts.append(text)
                                        filenames.append(f"{filename}")
                                        text_output_path = os.path.join(text_output_dir, f"{base}_{k+1}.txt")
                                        with open(text_output_path, 'w') as f:
                                            f.write(text)
                                    cidlist = list(filter(lambda a: a != i, cidlist))
                                if not cidlist:
                                    break
                        else:        
                            text = ocr_image(output_path)
                            extracted_texts.append(text)
                            filenames.append(f"{base}_{file_count[base]}{ext}")
                            text_output_path = os.path.join(text_output_dir, f"{base}_{file_count[base]}.txt")
                            with open(text_output_path, 'w') as f:
                                f.write(text)
                else:
                    continue
 
        for task in pdf_tasks:
            texts = task.result()
            extracted_texts.extend(texts)
 
    return text_output_dir, filenames, cid_to_filename

def shingle_gen(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k + 1):
        shingle_set.append(text[i:i + k])
    return set(shingle_set)

def process_files(directory, threshold):
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    
    minhashes = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
 
    for file in file_paths:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            shingles = shingle_gen(text, 5)
            minhash = MinHash(num_perm=128)
            for shingle in shingles:
                minhash.update(shingle.encode('utf-8'))
            minhashes[file] = minhash  
            lsh.insert(file, minhash)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue
 
    duplicates = []
 
    for file1 in minhashes:
        minhash1 = minhashes[file1]
        result = lsh.query(minhash1)
        for file2 in result:
            if file1 < file2:
                minhash2 = minhashes[file2]  
                similarity_score = minhash1.jaccard(minhash2)
                if similarity_score >= threshold:
                    duplicates.append({
                        'file1': file1,
                        'file2': file2,
                        'duplicate': True,
                        'score': similarity_score
                    })
 
    return {
        'duplicates': duplicates,
        'files': file_paths
    }

# Flask routes

@app.route('/', methods=['POST'])
def upload_file():
    print("Received request")  # Add a print statement to indicate that a request was received
    
    if 'file' not in request.files:
        print("No file part in request")  # Print a message if there's no file part in the request
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("No selected file")  # Print a message if no file was selected
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        print("File uploaded successfully")  # Print a message if the file was uploaded successfully
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        start_time = time.time()
        text_output_dir, filenames, cid_to_filename = extract_attachments(filepath, OUTPUT_FOLDER)
        print("Attachments extracted successfully")  # Print a message if attachments were extracted successfully
        
        threshold = 0.8
        result = process_files(text_output_dir, threshold)
        print("Result :",result)
        print("Files processed successfully")  # Print a message if files were processed successfully
        end_time = time.time()
        
        # Return the processed data as a JSON response
        return jsonify({
            'text_output_dir': text_output_dir,
            'filenames': filenames,
            'cid_to_filename': cid_to_filename,
            'duplicates': result['duplicates'],
            'files': result['files'],
            'execution_time': f"{end_time - start_time:.2f} seconds"
        })
    
    else:
        print("Unsupported file type")  # Print a message if the file type is not supported
        return jsonify({'error': 'Unsupported file type'}), 400

  

if __name__ == "__main__":
    app.run(debug=True)


        

