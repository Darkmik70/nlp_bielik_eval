import json
import chardet  # Install with `pip install chardet` if not already installed

# Function to extract the best score
def extract_best_score(data):
    if not data:
        return None
    
    # Find the entry with the highest f1_score
    best_entry = max(data, key=lambda x: x['f1_score'])
    return best_entry

# Load JSON data from a file with encoding detection
def load_json_file(file_path):
    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)['encoding']
        
        # Read the file using the detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

# Main function
if __name__ == "__main__":
    # Replace 'data.json' with your JSON file path
    file_path = ""
    
    # Load the JSON file
    data = load_json_file(file_path)
    
    if data:
        # Extract the best score
        best_score_entry = extract_best_score(data)
        
        # Display the best score entry
        if best_score_entry:
            print("Best Score Entry:")
            print(json.dumps(best_score_entry, indent=4, ensure_ascii=False))
        else:
            print("No data available.")
    else:
        print("Failed to load data from the file.")
