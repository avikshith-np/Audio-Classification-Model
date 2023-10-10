# Audio Genre Classification

This project focuses on classifying music genres using Convolutional Neural Networks (CNNs) with audio features. It uses the GTZAN dataset for music samples and Common Voice dataset for additional data.

## Introduction

Music genre classification is a fundamental task in the field of audio analysis and machine learning. This project demonstrates how to build a CNN-based model to automatically classify music tracks into predefined genres. The model extracts audio features, such as Mel-frequency cepstral coefficients (MFCCs), to represent audio data and then uses these features for classification.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Python
- Conda (for managing environments, optional but recommended)
- Libraries: librosa, numpy, scikit-learn, keras
- GTZAN dataset (you can obtain it online)
- Common Voice dataset (you can obtain it online)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/avikshith-np/Audio-Classification-Model.git
   
2. Create and activate a Conda environment (optional but recommended):
   ```bash
   conda create -n audio-classification-env python=3.9
   conda activate audio-classification-env
   
4. Install the required Python packages:
   ```bash
   pip install librosa numpy tensorflow

5. Download the GTZAN dataset and Common Voice dataset and update the dataset paths in the code accordingly.
