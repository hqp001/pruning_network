import os
import gzip

def gzip_file(input_file_path, output_gzip_path):
    try:
        os.makedirs(os.path.dirname(output_gzip_path), exist_ok=True)

        # Open the input file in binary read mode
        with open(input_file_path, 'rb') as input_file:
            # Open the output file in gzip write mode
            with gzip.open(output_gzip_path, 'wb') as output_file:
                # Read from the input file and write to the gzip file
                output_file.writelines(input_file)


        print(f"File {input_file_path} has been compressed and saved as {output_gzip_path}")
    except Exception as e:
        print(f"An error occurred while compressing {input_file_path}: {e}")

FOLDER = "./vnncomp2022_results/dense_0"
# Get the current directory
current_directory = f"{FOLDER}/mnist_fc"

print(current_directory)
# List all files in the current directory
files_in_directory = os.listdir(f"{current_directory}")

print(files_in_directory)

# Filter out directories and keep only files
files = [f for f in files_in_directory if os.path.isfile(os.path.join(current_directory, f))]

# Gzip each file
for file in files:
    print(file)
    input_file_path = os.path.join(current_directory, file)
    output_gzip_path = file + '.gz'
    gzip_file(input_file_path, f"./{FOLDER}/mnist_fc/{output_gzip_path}")

