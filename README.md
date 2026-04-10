# Neural Text Summarization: Amazon Fine Food Reviews

This project implements an abstractive text summarization system using a Sequence-to-Sequence (Seq2Seq) architecture with LSTM layers and an Attention-like mechanism. The model is trained on the Amazon Fine Food Reviews dataset to generate concise summaries from descriptive customer reviews.

## Project Overview

The objective is to transform long-form customer feedback into short, meaningful summaries. Unlike extractive summarization (which picks existing sentences), this abstractive approach generates new sentences that capture the essence of the input text.

### Key Features
- **Data Preprocessing**: Handles HTML tags, contractions expansion, stopword removal, and text normalization.
- **Model Architecture**: 
  - 3-layer stacked LSTM Encoder.
  - LSTM Decoder with Attention mechanism.
  - Sparse Categorical Cross-Entropy loss.
- **Inference Pipeline**: Real-time generation of summaries using a greedy search decoder.
- **Performance Evaluation**: Uses ROUGE metrics to assess the quality of generated summaries against references.

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib
- NLTK, BeautifulSoup4

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-summarization.git
   cd text-summarization
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy nltk beautifulsoup4 lxml matplotlib
   ```

## Dataset
The project uses the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. Ensure `Reviews.csv` is available in your data directory before running the notebook.

## Model Summary
The model utilizes an Encoder-Decoder framework where the encoder compresses the input review into a'context vector' and the decoder expands it into a summary, token by token.
