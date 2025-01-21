import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = '../scored_results/detailed_and_averaged_results.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Convert columns to numeric, coercing invalid values to NaN
data['F1 Score'] = pd.to_numeric(data['F1 Score'], errors='coerce')
data['Similarity Score'] = pd.to_numeric(data['Similarity Score'], errors='coerce')

# Drop rows with NaN values
data = data.dropna(subset=['F1 Score', 'Similarity Score'])

# Create a 'Run' column: Each 100 steps is a different run
data['Run'] = (data.index // 100) + 1

# Map runs to model names
run_labels = {
    1: "Bielik Run 1",
    2: "Bielik Run 2",
    3: "Bielik Run 3",
    4: "GPT-3.5 Run 1",
    5: "GPT-3.5 Run 2",
    6: "GPT-3.5 Run 3",
    7: "GPT-4o-mini Run 1",
    8: "GPT-4o-mini Run 2",
    9: "GPT-4o-mini Run 3"
}
data['Run Name'] = data['Run'].map(run_labels)

# Extract model names from run labels
data['Model'] = data['Run Name'].str.split().str[0]

# Group by 'Run Name' and calculate averages for F1 and similarity scores
run_average_scores = data.groupby('Run Name').agg({
    'F1 Score': 'mean',
    'Similarity Score': 'mean'
}).reset_index()

# Group by 'Model' to calculate overall averages for F1 and similarity scores
model_average_scores = data.groupby('Model').agg({
    'F1 Score': 'mean',
    'Similarity Score': 'mean'
}).reset_index()

# Prepare data for grouped bar plot
x = np.arange(len(run_average_scores['Run Name']))  # the label locations
width = 0.3  # the width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot F1 Score bars
f1_bars = ax.bar(x - width, run_average_scores['F1 Score'], width, label='F1 Score (Run)', alpha=0.8)

# Plot Similarity Score bars
similarity_bars = ax.bar(x, run_average_scores['Similarity Score'], width, label='Similarity Score (Run)', alpha=0.8)

# Add overall averages as a separate set of bars
model_x = np.arange(0, len(model_average_scores) * 3, step=3)  # Adjust spacing for models
overall_f1_bars = ax.bar(model_x + width, model_average_scores['F1 Score'], width, label='F1 Score (Overall)', alpha=0.6, color='gray')
overall_similarity_bars = ax.bar(model_x + 2 * width, model_average_scores['Similarity Score'], width, label='Similarity Score (Overall)', alpha=0.6, color='darkgray')

# Customize the plot
ax.set_title('Average F1 Score and Similarity Score with Overall Model Averages', fontsize=16)
ax.set_xlabel('Run', fontsize=14)
ax.set_ylabel('Average Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(run_average_scores['Run Name'], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels to the bars
for bars in [f1_bars, similarity_bars, overall_f1_bars, overall_similarity_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,  # Add a small offset above the bar
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# Save the plot as a PNG file
output_path = '../scored_results/average_scores_with_overall_averages.png'  # Replace with your desired output path
plt.tight_layout()
plt.savefig(output_path, format='png', dpi=300)

# Show the plot
plt.show()
cdcd 