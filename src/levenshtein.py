import json
import os
from rapidfuzz.distance import Levenshtein
import chardet


def load_json(file_path):
    """Load a JSON file with automatic encoding detection."""
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
        print(f"Error loading file {file_path}: {e}")
        return None


def calculate_similarity(prediction, true_answer):
    """
    Calculate text similarity using normalized Levenshtein distance.
    Returns a similarity score between 0 and 1.
    """
    if not prediction or not true_answer:
        return 0.0
    dist = Levenshtein.distance(prediction.lower(), true_answer.lower())
    max_len = max(len(prediction), len(true_answer))
    return 1 - (dist / max_len)


def assess_text_similarity(data):
    """
    Assess the text similarity for each question in the dataset.
    Returns the dataset with added similarity scores.
    """
    scored_data = []
    total_similarity = 0.0
    count = 0

    for entry in data:
        prediction = entry.get("prediction", "")
        true_answers = entry.get("true_answers", [])

        if true_answers:
            true_answer = true_answers[0]
            similarity = calculate_similarity(prediction, true_answer)
            total_similarity += similarity
            count += 1

            entry['similarity_score'] = similarity
            scored_data.append(entry)

            print(f"Question: {entry['question']}")
            print(f"Prediction: {prediction}")
            print(f"True Answer: {true_answer}")
            print(f"Similarity: {similarity:.2f}")
            print("-" * 50)

    average_similarity = total_similarity / count if count > 0 else 0.0
    return scored_data, average_similarity


def save_results(output_path, data):
    """Save scored data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def process_json_files(input_directory, output_directory):
    """
    Process all JSON files in a directory, calculate text similarity, and save results.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".json"):
            input_path = os.path.join(input_directory, file_name)
            output_path = os.path.join(output_directory, f"scored_{file_name}")

            print(f"Processing file: {file_name}")

            try:
                data = load_json(input_path)
                if data is not None:
                    scored_data, avg_similarity = assess_text_similarity(data)
                    save_results(output_path, {"results": scored_data, "average_similarity": avg_similarity})
                    print(f"Results saved to {output_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    # Directories for input and output files
    input_directory = ""  # Update this path to your input JSON file directory
    output_directory = "./scored_results"  # Update this path to your output JSON directory

    # Ensure the input directory exists
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' does not exist. Please update the path.")
    else:
        process_json_files(input_directory, output_directory)
