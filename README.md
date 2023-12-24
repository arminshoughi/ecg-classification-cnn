# ECG Signal Classification with Deep Learning

## Overview

Welcome to the repository for the implementation of our paper on accurate Electrocardiogram (ECG) signal classification using deep learning. ECG signals play a vital role in providing crucial cardiovascular information for medical practitioners. Manual analysis of these signals is intricate and time-consuming, requiring specific skills. The challenges posed by noise, signal rigidity, and irregular heartbeats make it essential to employ advanced techniques for accurate classification.

## Approach

Our approach leverages a Convolutional Neural Network (CNN), discrete wavelet transformation with db2 mother wavelet, and the Synthetic Minority Over-sampling Technique (SMOTE). We applied this methodology to the MIT-BIH dataset, adhering to the Association for the Advancement of Medical Instrumentation (AAMI) standards. The aim is to enhance the accuracy of ECG signal classifications, particularly for cardiovascular diseases (CVDs), a leading global cause of mortality.

## Results

After training for 50 epochs, with each epoch taking 39 seconds, our approach achieved remarkable accuracy:
- Category F: 99.71%
- Category N: 98.69%
- Category S: 99.45%
- Category V: 99.33%
- Category Q: 99.82%

These results demonstrate the effectiveness of our model in accurately classifying ECG signals, making it a potential clinical auxiliary diagnostic tool.

## Usage

To use the code, clone the repository:

```bash
git clone https://gitlab.com/arminshoughi/ecg-classification-cnn
cd ecg-classification-cnn
pip install -r requirements.txt
python EcgClassification.py
```

Follow the installation steps and guidelines in the codebase to replicate our experiments and apply the model to your own datasets.

## Evaluation

For a detailed understanding of our methodology and evaluation results, please refer to our published paper. The complete article is available [here](https://link.springer.com/article/10.1007/s00607-023-01243-0).

## Contact

For any inquiries or collaboration opportunities, feel free to contact the project maintainer, Armin Shoughi, via the repository [issues](https://github.com/arminshoughi/ecg-classification-cnn/issues).

Thank you for your interest in our work!
