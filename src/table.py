import os
import json
import csv
from chardet import detect


def load_json(file_path):
    """Load a JSON file with automatic encoding detection and validation."""
    try:
        # Detect file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding = detect(raw_data)['encoding']
        
        # Read the file using the detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            data = json.load(file)

        # Ensure the data is a list of dictionaries
        if isinstance(data, dict) and "results" in data:
            return data["results"], data.get("average_similarity", 0)
        else:
            raise ValueError(f"Invalid data structure in {file_path}. Expected a dictionary with 'results'.")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, 0


def calculate_averages(model_data):
    """Calculate averages for Exact Match, F1 Score, and Similarity Score."""
    total_em = 0.0
    total_f1 = 0.0
    total_similarity = 0.0
    num_entries = 0

    for entry in model_data:
        total_em += entry.get('exact_match', 0)
        total_f1 += entry.get('f1_score', 0)
        total_similarity += entry.get('similarity_score', 0)
        num_entries += 1

    return {
        "average_exact_match": total_em / num_entries if num_entries > 0 else 0,
        "average_f1_score": total_f1 / num_entries if num_entries > 0 else 0,
        "average_similarity": total_similarity / num_entries if num_entries > 0 else 0,
        "num_entries": num_entries
    }


def process_model_runs(directory):
    """Process all JSON files for a specific model and calculate metrics."""
    results = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)
            print(f"Processing file: {file_name}")
            try:
                data, avg_similarity = load_json(file_path)
                if data is None:
                    continue  # Skip invalid files
                model_name = file_name.split('_')[1]  # Assumes format like "scored_Model_Run.json"
                if model_name not in results:
                    results[model_name] = {'entries': [], 'average_similarity': []}
                results[model_name]['entries'].extend(data)  # Add all entries from this run
                results[model_name]['average_similarity'].append(avg_similarity)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return results


def save_to_csv(output_file, results):
    """Save detailed results and averaged results to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header for detailed results
        writer.writerow(["Model", "Question", "Exact Match", "F1 Score", "Similarity Score"])

        # Write detailed results for each model
        for model, model_data in results.items():
            for entry in model_data['entries']:
                writer.writerow([
                    model,
                    entry.get("question", ""),
                    entry.get("exact_match", 0),
                    entry.get("f1_score", 0),
                    entry.get("similarity_score", 0)
                ])
        
        # Add a blank line for separation
        writer.writerow([])

        # Write header for averages
        writer.writerow(["Model", "Average Exact Match (%)", "Average F1 Score (%)", "Average Similarity (%)"])

        # Write averaged results
        for model, model_data in results.items():
            averages = calculate_averages(model_data['entries'])
            overall_avg_similarity = sum(model_data['average_similarity']) / len(model_data['average_similarity']) if model_data['average_similarity'] else 0
            writer.writerow([
                model,
                averages["average_exact_match"] * 100,
                averages["average_f1_score"] * 100,
                overall_avg_similarity * 100
            ])


if __name__ == "__main__":
    # Input directory containing JSON files
    input_directory = "/home/michal/workspaces/nlp_bielik_eval/scored_results"  # Update to your input directory
    output_csv = "./detailed_and_averaged_results.csv"  # Output CSV file path

    # Process model runs and calculate averages
    model_data = process_model_runs(input_directory)

    # Save results to CSV
    save_to_csv(output_csv, model_data)
    print(f"Results saved to {output_csv}")
