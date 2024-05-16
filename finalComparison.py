import re
from difflib import SequenceMatcher
import time


def extract_numerical_data(text):
    pattern_numbers = r"\d+(?:\.\d+)?(?=$|\s)"  
    return re.findall(pattern_numbers, text)


def extract_total_value(text):
    total_match = re.search(r'total\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if total_match:
        total_value = float(total_match.group(1))
        return total_value
    else:
        return None


def is_bill_identical(text_file1, text_file2):
    with open(text_file1, 'r') as f:
        text1 = f.read()
    with open(text_file2, 'r') as f:
        text2 = f.read()

    total_value1 = extract_total_value(text1)
    total_value2 = extract_total_value(text2)

    if total_value1 is not None and total_value2 is not None:
        if abs(total_value1 !=total_value2) :  # Adjust tolerance as needed
            # Total values are different, skip further comparisons
            return False

    # Compare numerical data for close matches (adjust tolerance as needed)
    tolerance = 0.1  # Allow for a 10% difference in numerical values
    numerical_data1 = extract_numerical_data(text1)
    numerical_data2 = extract_numerical_data(text2)
    num_data_match = all(abs(float(num1) - float(num2)) <= tolerance * max(float(num1), float(num2))
                         for num1, num2 in zip(numerical_data1, numerical_data2))

    # Compare text similarity
    text_matcher = SequenceMatcher(None, text1, text2)
    text_ratio = text_matcher.quick_ratio()

    # Return True only if all comparisons pass
    return text_ratio >= 0.95 and num_data_match


def compare_all_bills(file_paths):
    identical_files = []
    unique_files = set(file_paths)

    for i, file_path1 in enumerate(file_paths):
        for file_path2 in file_paths[i+1:]:
            if is_bill_identical(file_path1, file_path2):
                identical_files.append((file_path1, file_path2))
                unique_files.discard(file_path1)
                unique_files.discard(file_path2)

    if identical_files:
        print("Identical files:")
        for file1, file2 in identical_files:
            print(f"{file1} and {file2}")
    else:
        print("All files are unique")
    
    if unique_files:
        print("\nUnique files:")
        for file_path in unique_files:
            print(file_path)

# Sample file paths (replace with your actual list)
file_paths = [
    "/Users/I748992/Downloads/emltotxt/attachments/8a45513f-a179-4fd3-ad5a-40558eca36ea-1715224792372/holidayinn_1_1.txt",
    "//Users/I748992/Downloads/emltotxt/attachments/e69d3fc1-a8c4-4caf-8f30-2497a9c8c8f5-1715247695362/PNG REceipt_1.txt",
    "/Users/I748992/Downloads/emltotxt/attachments/Receipts_531618793/aldi_20220505_1_2.txt",
    "/Users/I748992/Downloads/emltotxt/attachments/e69d3fc1-a8c4-4caf-8f30-2497a9c8c8f5-1715247695362/PNG REceipt_1.txt",
    "/Users/I748992/Downloads/emltotxt/attachments/8a45513f-a179-4fd3-ad5a-40558eca36ea-1715224792372/holidayinn_2_2.txt"
]  # Add your file paths here

start_time = time.time()
compare_all_bills(file_paths)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)
