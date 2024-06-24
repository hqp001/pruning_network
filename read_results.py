import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Model state dictionary checker and saver.")
parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
parser.add_argument('--file', type=str, required=True, help="Name of file to read")
args = parser.parse_args()

# Read the CSV file
file_path = f'./results/seed_{args.seed}/{args.file}.csv'  # Update with the actual file path
df = pd.read_csv(file_path)

df_true = df[df['train_first'] == True]
df_false = df[df['train_first'] == False]

# Check the unique sparsity values in the dataset
sparsity_values = df['sparsity'].unique()

sparsity_values.sort()

# Create a dictionary to hold the dataframes for each sparsity value
sparsity_data_true = {sparsity: df_true[df_true['sparsity'] == sparsity] for sparsity in sparsity_values}
sparsity_data_false = {sparsity: df_false[df_false['sparsity'] == sparsity] for sparsity in sparsity_values}


# Find the count of valid images and the average runtime for valid images
valid_image_counts = {}
average_runtime_valid_image = {}
average_runtime_total = {}
total_rows_per_sparsity = {}
total_rows_per_sparsity_false = {}
valid_image_counts_false = {}

for sparsity, data in sparsity_data_true.items():

    total_rows_per_sparsity[sparsity] = data.shape[0]

    valid_data = data[data['valid_image'] == True]
    valid_count = valid_data.shape[0]
    valid_image_counts[sparsity] = valid_count

    avg_runtime = data['runtime'].mean()
    average_runtime_total[sparsity] = avg_runtime

    if valid_count > 0:
        average_runtime = valid_data['runtime'].mean()
        average_runtime_valid_image[sparsity] = average_runtime
    else:
        average_runtime_valid_image[sparsity] = None

for sparsity, data in sparsity_data_false.items():

    total_rows_per_sparsity_false[sparsity] = data.shape[0]

    valid_data = data[data['valid_image'] == True]
    valid_count = valid_data.shape[0]
    valid_image_counts_false[sparsity] = valid_count


# Display the results
for sparsity in sparsity_values:
    if (sparsity > 0):
        print(f'Sparsity {sparsity}:')
        print(f'  Total valid images: {valid_image_counts[sparsity]} / {total_rows_per_sparsity[sparsity]}')
        print(f'  Ablation experiments: {valid_image_counts_false[sparsity]} / {total_rows_per_sparsity_false[sparsity]}')
    else:
        print(f'Dense:')
        total_rows = total_rows_per_sparsity[sparsity] + total_rows_per_sparsity_false[sparsity]
        print(f'  Total valid images: {valid_image_counts[sparsity]} / {total_rows}')
    print(f'  Average runtime: {average_runtime_total[sparsity]}')
    if average_runtime_valid_image[sparsity] is not None:
        print(f'  Average runtime for valid images: {average_runtime_valid_image[sparsity]:.4f}')
    else:
        print(f'  No valid images')

    print()

