# Heart Anomaly Detection 🫀

[GitHub]([https://github.com/Kraken2003/heart-anomaly/tree/main])

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Pre-processing](#preprocessing)
- [Model Information](#model)
- [Usage](#usage)
- [License](#license)

## Introduction

📈 This repository contains the code and resources for predicting heart anomalies in patients using their ECG data. We implemented deep neural network models, including Long Short-Term Memory (LSTM) and self-attention networks, to process time series ECG data and achieved an accuracy of 85%+ on the test dataset.
Cardiac arrhythmias, irregular heart rhythms, can be indicative of underlying health issues and pose a risk to patients' well-being. By harnessing the potential of deep neural networks, we've developed a robust model capable of analyzing ECG time series data and classifying different types of arrhythmias. Achieving an accuracy of over 85% on our test dataset showcases the potential impact of this technology in the medical field.

## Dataset

📊 The dataset used for this project is the [ECG Arrhythmia Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/) from PhysioNet. It consists of 105 records, each containing 12-lead ECG signals and annotations for various arrhythmias and cardiac events.
The dataset acts as a treasure trove of real-world ECG data, enabling us to simulate and explore a diverse range of cardiac scenarios. Each record encapsulates valuable information about the heart's electrical activity, providing a dynamic and intricate representation of cardiac function.

### Preprocessing

⚙️ Converting raw data into a format suitable for deep learning is a crucial step in our workflow. To ensure accurate and efficient model training, we performed the following preprocessing steps:

1. Data Conversion: We meticulously extracted and transformed data from the original .mat and .hea files, converting them into a structured CSV format. This enabled us to organize the data into coherent sequences for analysis.

2. Normalization: Achieving consistent data scaling is vital for training neural networks. We applied normalization techniques to the ECG signals, ensuring that the data falls within a standardized range while preserving its underlying patterns.

3. Batch Processing: Efficient training of deep neural networks often requires batch processing. By dividing the data into manageable batches and associating them with corresponding labels, we facilitated seamless integration into our chosen neural network architectures.

## Model Information

🧠 Our exploration into accurate arrhythmia detection led us to experiment with two distinctive recurrent neural network architectures:

1. LSTM (Long Short-Term Memory): The LSTM architecture has demonstrated remarkable success in capturing temporal dependencies within sequential data. By employing LSTM units, our model is capable of learning intricate patterns present in the ECG time series, enhancing its ability to discern anomalies.

2. Self-Attention Networks: Self-attention mechanisms have proven invaluable in tasks that involve identifying salient features across sequences. We harnessed the power of self-attention to enable our model to focus on relevant segments within the ECG data, allowing for enhanced arrhythmia detection capabilities.

The utilization of different window sizes to accommodate the default CSV data interval of 10 seconds, divided into 5000 time steps, showcases our dedication to tailoring our model to the unique characteristics of ECG data.

## Usage

🚀 Follow these steps to replicate and build upon our work:

1. Clone this repository: `git clone https://github.com/Kraken2003/heart-anomaly/tree/main`
2. Navigate to the project directory: `cd heart-anomaly`
3. Run "pip install requirements.txt"
4. Run make output_leads notebook.
5. Adjust hyperparameters, network architecture, and preprocessing steps as needed.
6. Train and evaluate the models on your own data or the ECG Arrhythmia Database.
7. Share your findings, improvements, and contributions with the community by submitting pull requests!

## License

📜 This project is licensed under the [MIT License](LICENSE).
