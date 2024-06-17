import pickle
import os

# Get directory to the output folder
path_out = os.path.join(os.getcwd(), '../output')
out_files = os.listdir(path_out)

# Get all pickle files in the output folder
pickle_paths = []
for file in out_files:
    if file.endswith('.pickle'):
        file_output = os.path.join(path_out, file)
        pickle_paths.append(file_output)

# Get all .out files in the output folder
out_paths = []
for file in out_files:
    if file.endswith('.out'):
        file_output = os.path.join(path_out, file)
        out_paths.append(file_output)

# Read the content in each pickle file
for file_path in pickle_paths:
    print(file_path)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(data)
print("\n\n")

# Read the content in each .out file
for file_path in out_paths:
    print(file_path)

    with open(file_path, 'r') as file:
        content = file.read()
        print(content)
        # process the content of the .out file as needed
print("\n\n")
    