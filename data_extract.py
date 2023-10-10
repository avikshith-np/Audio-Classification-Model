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