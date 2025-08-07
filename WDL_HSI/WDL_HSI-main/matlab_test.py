import scipy.io

# Replace 'your_file.mat' with the path to your .mat file
mat_data = scipy.io.loadmat('pavia_gt.mat')

# Print the keys in the loaded .mat file
print(mat_data.keys())

data = mat_data['pavia_gt']
print(type(data))
print(data.shape)
print(data.dtype)

# Access a specific variable (example: 'my_variable')
# my_var = mat_data['my_variable']