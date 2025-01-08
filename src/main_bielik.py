#!/home/michal/miniconda3/bin/python3
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the dataset
dataset = load_dataset("clarin-pl/poquad")

# Select a subset for evaluation (e.g., first 100 samples from the validation set)
eval_subset = dataset['validation'].select(range(100))

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-11B-v2.3-Instruct-GPTQ")
model = AutoModelForCausalLM.from_pretrained("speakleash/Bielik-11B-v2.3-Instruct-GPTQ",device_map="cuda:0")
# Move model to GPU if available
device = torch.device("cuda:0")
model.to(device)

# Function to generate predictions
def generate_answer(question, context):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():  # No need for gradients during inference
        outputs = model.generate(**inputs, max_new_tokens=30) #short answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Evaluation function
def evaluate(dataset):
    em_total = 0
    f1_total = 0
    results = []  # List to store results for saving
    
    for example in dataset:
        context = example['context']
        question = example['question']
        true_answers = example['answers']['text']
        try:
            prediction = generate_answer(question, context)
        except Exception as e:
            print(f"Error generating answer for question: {question}\n{e}")
            prediction = ""
        
        # Compute Exact Match and F1 scores
        em_score = max(compute_exact(prediction, ans) for ans in true_answers)
        f1_score = max(compute_f1(prediction, ans) for ans in true_answers)
        
        em_total += em_score
        f1_total += f1_score
        
        # Store result
        results.append({
            "question": question,
            "context": context,
            "true_answers": true_answers,
            "prediction": prediction,
            "exact_match": em_score,
            "f1_score": f1_score
        })
    
    # Compute overall metrics
    em_avg = em_total / len(dataset) * 100
    f1_avg = f1_total / len(dataset) * 100
    
    return em_avg, f1_avg, results

# Helper functions to compute Exact Match and F1 scores
def compute_exact(a_pred, a_true):
    return int(a_pred.strip() == a_true.strip())

def compute_f1(prediction, true_answer):
    # Example logic for computing precision and recall
    true_positives = len(set(prediction) & set(true_answer))
    false_positives = len(set(prediction) - set(true_answer))
    false_negatives = len(set(true_answer) - set(prediction))

    if true_positives == 0:
        prec = 0
        rec = 0
    else:
        prec = true_positives / (true_positives + false_positives)
        rec = true_positives / (true_positives + false_negatives)

    # Handle the case where both precision and recall are zero
    if (prec + rec) == 0:
        return 0  # Avoid division by zero by returning F1 score as 0
    
    return 2 * (prec * rec) / (prec + rec)


# def compute_f1(a_pred, a_true):
#     pred_tokens = a_pred.strip().split()
#     true_tokens = a_true.strip().split()
#     if not pred_tokens or not true_tokens:
#         return 0
#     common = set(pred_tokens) & set(true_tokens)
#     prec = len(common) / len(pred_tokens)
#     rec = len(common) / len(true_tokens)
#     return 2 * (prec * rec) / (prec + rec)

# Run evaluation
em, f1, results = evaluate(eval_subset)
print(f"Exact Match: {em:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save results to JSON file
output_file = "evaluation_results.json"
with open(output_file, "w", encoding="utf-16") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"Results saved to {output_file}")
