from openai import OpenAI
import json
import time
from datasets import load_dataset

# Set API key (use environment variable for security)
client = OpenAI(api_key="")

# Generate answer using GPT-4
def generate_answer_gpt4(question, context):
    try:
        prompt = f"Question: {question}\nContext: {context}\nProvide a short and concise answer:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant specializing in answering questions concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer for question: {question}\n{e}")
        return ""

# Evaluate GPT-4
def evaluate_gpt4(dataset):
    em_total = 0
    f1_total = 0
    results = []

    for example in dataset:
        context = example["context"]
        question = example["question"]
        true_answers = example["answers"]["text"]
        
        prediction = generate_answer_gpt4(question, context)

        em_score = max(compute_exact(prediction, ans) for ans in true_answers)
        f1_score = max(compute_f1(prediction, ans) for ans in true_answers)

        em_total += em_score
        f1_total += f1_score

        results.append({
            "question": question,
            "context": context,
            "true_answers": true_answers,
            "prediction": prediction,
            "exact_match": em_score,
            "f1_score": f1_score,
        })

    em_avg = em_total / len(dataset) * 100
    f1_avg = f1_total / len(dataset) * 100

    return em_avg, f1_avg, results

# Metrics
def compute_exact(a_pred, a_true):
    return int(a_pred.strip() == a_true.strip())

def compute_f1(a_pred, a_true):
    pred_tokens = a_pred.strip().split()
    true_tokens = a_true.strip().split()
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(true_tokens)
    return 2 * (prec * rec) / (prec + rec)

# Load dataset
dataset = load_dataset("clarin-pl/poquad")["validation"].select(range(100))

# Run evaluation
print("Evaluating GPT-4...")
start_time = time.time()

em, f1, results = evaluate_gpt4(dataset)

# Save results
output_file = "gpt3.5_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

end_time = time.time()
print("Results for GPT-3.5:")
print(f"  Exact Match: {em:.2f}")
print(f"  F1 Score: {f1:.2f}")
print(f"  Runtime: {end_time - start_time:.2f} seconds")
print(f"  Saved results to {output_file}")
