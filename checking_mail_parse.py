import os
import warnings
import time
import difflib
from concurrent.futures import ThreadPoolExecutor
from email import message_from_binary_file
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)

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
    return img_tags

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
    cid_count = {}
    cid_to_filename = {}
    cidlist=[]
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
                elif (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')) and (len(img_tags) < 1)):
                    text = ocr_image(output_path)
                    extracted_texts.append(text)
                    filenames.append(f"{base}_{file_count[base]}{ext}")
                    text_output_path = os.path.splitext(output_path)[0] + f"_{file_count[base]}.txt"
                    with open(text_output_path, 'w') as f:
                        f.write(text)
                else:
                    if(len(cidlist) > 0):
                        data = part.get_payload(decode=True)
                        for i in cidlist:
                            if i in content_id:
                                cid_to_filename[i] = filename
                                print(f"Processing CID: {i}")
                                for k in range(cid_count[i]):
                                    text = ocr_image(output_path)
                                    extracted_texts.append(text)
                                    filenames.append(f"{filename}")
                                    text_output_path = os.path.splitext(output_path)[0] + f"_{k+1}.txt"
                                    with open(text_output_path, 'w') as f:
                                        f.write(text)
                                    print(f"Done CID: {i}, count: {k+1}")
                                cidlist = list(filter(lambda a: a != i, cidlist))
                                print(f"Remaining CIDs: {len(cidlist)}")
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

# Function to compute detailed string similarity
def compute_detailed_similarity(text1, text2):
    sequence_matcher = difflib.SequenceMatcher(None, text1, text2)
    return sequence_matcher.ratio()

# Main function to extract texts, compute embeddings, cluster texts, and display results
def main():
    start_time = time.time()
    # Load Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    eml_file_path = '/Users/I748970/Downloads/emltotxt/testfiles/testing8.eml'
    output_dir = '/Users/I748970/Downloads/emltotxt/attachments'
    # Extract texts and filenames from email attachments
    extracted_texts, filenames, cid_to_filename = extract_attachments(eml_file_path, output_dir)
    
    # Debug extraction
    if not extracted_texts:
        print("No texts were extracted.")
        return
    
    # Get embeddings
    embeddings = get_embeddings(extracted_texts, model)
    
    # DBSCAN clustering
    eps = 0.3  # Adjust this parameter as needed
    min_samples = 2  # Adjust this parameter as needed
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(embeddings)
    
    # Compute similarity scores
    similarity_matrix = cosine_similarity(embeddings)
    
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
                    semantic_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                    detailed_similarity = compute_detailed_similarity(text1, text2)
                    
                    print(f"\nSemantic Similarity Score between {file1} and {file2}: {semantic_similarity}")
                    if(semantic_similarity>0.8):
                        print(f"Detailed Similarity Score between {file1} and {file2}: {detailed_similarity}")
                    
                    if semantic_similarity > 0.8:
                        if detailed_similarity < 1.0:
                            print(f"\nPrinting differences for similar files:")
                            print_differences(text1, text2, file1, file2)
                        else:
                            print("\n !!! DUPLICATES PRESENT !!!")
                    else:
                        print("\n *** NO DUPLICATES ***")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
