import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from concurrent.futures import ThreadPoolExecutor
import json

# Define the genres and their corresponding labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
num_classes = len(genres)
count = 0

# Define a function to extract audio features (e.g., MFCCs)
def extract_features(audio_file):
    try:
        audio, _ = librosa.load(audio_file, sr=44100)  # Set the sample rate here
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13), axis=1)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {str(e)}")
        return None

# Create empty lists to store data and labels
data = []
labels = []

# Load GTZAN dataset
gtzan_dir = 'Music/genres_original'  # Update with the actual path to your GTZAN dataset
for genre_label, genre in enumerate(genres):
    genre_dir = os.path.join(gtzan_dir, genre)
    for filename in os.listdir(genre_dir):
        audio_file = os.path.join(genre_dir, filename)
        features = extract_features(audio_file)
        if features is not None:
            data.append(features)
            labels.append(genre_label)
            count = count + 1
            print(f"Processed {count}/1000 music files")

print("GTZAN Feature Extraction Completed")

# Define a function to load Common Voice dataset using multithreading
def load_common_voice_data(audio_files):
    results = []
    
    def process_audio_file(audio_file):
        nonlocal counter
        features = extract_features(audio_file)
        if features is not None:
            results.append(features)
        counter += 1
        print(f"Processed {counter}/{len(audio_files)} audio files")
    
    counter = 0
    
    with ThreadPoolExecutor(max_workers=22) as executor:
        executor.map(process_audio_file, audio_files)
    
    return results

# Load Common Voice dataset (negative samples) using multithreading
common_voice_dir = 'Speech/cv-corpus-15.0-2023-09-08/en/clips'  # Update with the actual path to your Common Voice dataset
#common_voice_files = [os.path.join(common_voice_dir, filename) for filename in os.listdir(common_voice_dir)]
common_voice_files = [os.path.join(common_voice_dir, filename) for filename in os.listdir(common_voice_dir)[:20000]]
common_voice_data = load_common_voice_data(common_voice_files)
print("Common Voice Feature Extraction Completed")

# Save common_voice_data to JSON file
common_voice_data_list = [d.tolist() for d in common_voice_data]
with open('common_voice_data.json', 'w', encoding='utf-8') as f:
    json.dump(common_voice_data_list, f, ensure_ascii=False, indent=4)
print("Json Data Dumped to common_voice_data.json")

# Assign labels to Common Voice data
common_voice_labels = [num_classes] * len(common_voice_data)

# Extend the data and labels lists with Common Voice data
data.extend(common_voice_data)
labels.extend(common_voice_labels)

# Save data and labels to JSON files
data_list = [d.tolist() for d in data]
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print("Saved data.json file")
print(labels)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

# Convert data to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape the data to be 4-dimensional (batch_size, n_mfcc, time_steps, 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

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
model.save('audio_classification_model.h5')
