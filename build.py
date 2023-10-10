import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load data from JSON files
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('common_voice_data.json', 'r', encoding='utf-8') as f:
    common_voice_data = json.load(f)

# Load labels
num_classes = len(data[0])  # Assuming all feature vectors have the same length
labels = [i for i in range(len(data))] + [num_classes] * len(common_voice_data)

# Convert data to numpy arrays
data = np.array(data)
common_voice_data = np.array(common_voice_data)

# Extend the data and labels lists with Common Voice data
data = np.vstack((data, common_voice_data))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

# Reshape the data to be 4-dimensional (batch_size, n_mfcc, time_steps, 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Convert labels to NumPy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

################# Labels Fix !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Ensure that labels are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Define the number of classes
num_genres = 10  # Number of music genres
num_classes = num_genres + 1  # Add 1 for the "Common Voice" class

# Check and update labels for both training and test data
for i in range(len(y_train)):
    if y_train[i] < 0 or y_train[i] >= num_classes:
        y_train[i] = num_genres  # Assign the "Common Voice" label to out-of-range values

for i in range(len(y_test)):
    if y_test[i] < 0 or y_test[i] >= num_classes:
        y_test[i] = num_genres  # Assign the "Common Voice" label to out-of-range values

# Check the unique label values again
unique_labels_train = np.unique(y_train)
unique_labels_test = np.unique(y_test)

# Ensure that labels are within the expected range
expected_labels = list(range(num_classes))
if not np.array_equal(unique_labels_train, expected_labels):
    print("Train labels still contain unexpected values.")
else:
    print("Train labels have been fixed.")

if not np.array_equal(unique_labels_test, expected_labels):
    print("Test labels still contain unexpected values.")
else:
    print("Test labels have been fixed.")

################# Labels Fix END !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Build a simple convolutional neural network (CNN) model
print("Building CNN model.........")
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes + 1, activation='softmax'))  # +1 for Common Voice label

# Compile the model
print("Compiling CNN model.........")
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Training CNN model.........")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Save the trained model
model.save('my_model.keras')
