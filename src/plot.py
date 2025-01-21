import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../scored_results/detailed_and_averaged_results.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert necessary columns to numeric, coercing errors to NaN
data['F1 Score'] = pd.to_numeric(data['F1 Score'], errors='coerce')
data['Similarity Score'] = pd.to_numeric(data['Similarity Score'], errors='coerce')

# Drop rows where numeric values are NaN
data = data.dropna(subset=['F1 Score', 'Similarity Score'])

# Create a group column for every 300 rows
data['Group'] = (data.index // 300) + 1  # Each group corresponds to 300 rows

# Group by 'Group' and calculate mean scores
averaged_data = data.groupby('Group')[['F1 Score', 'Similarity Score']].mean().reset_index()

# Replace group numbers with model names
model_names = {1: 'Bielik', 2: 'GPT3.5', 3: 'GPT 4o-mini'}
averaged_data['Group Name'] = averaged_data['Group'].map(model_names)

# Plot the averages as a column plot
plt.figure(figsize=(12, 6))

# Bar width for grouping columns
bar_width = 0.35
x = averaged_data['Group']

# Plot F1 Score bars
plt.bar(x - bar_width / 2, averaged_data['F1 Score'], width=bar_width, label='F1 Score Average', color='skyblue')

# Plot Similarity Score bars
plt.bar(x + bar_width / 2, averaged_data['Similarity Score'], width=bar_width, label='Similarity Score Average', color='salmon')

# Annotate the bars with their values
for i, row in averaged_data.iterrows():
    plt.text(row['Group'] - bar_width / 2, row['F1 Score'] + 0.01, f"{row['F1 Score']:.2f}", ha='center', fontsize=8)
    plt.text(row['Group'] + bar_width / 2, row['Similarity Score'] + 0.01, f"{row['Similarity Score']:.2f}", ha='center', fontsize=8)

# Customize the plot
plt.title('Average F1 Scores and Similarity Scores for Models', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.xticks(x, labels=averaged_data['Group Name'], fontsize=10)  # Add model names as x-ticks
plt.ylim(0, 1)  # Assuming scores are between 0 and 1
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and display the plot
plt.savefig("average_f1_similarity_scores_model_plot.png")
plt.show()
