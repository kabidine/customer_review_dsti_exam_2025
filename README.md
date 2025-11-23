# Movie Review Sentiment Analysis - DSTI Exam 2025

A sentiment analysis system for movie reviews using a fine-tuned DistilBERT model. The project includes model training, evaluation, and deployment through a FastAPI backend and Streamlit frontend.

## Overview

This project classifies movie reviews as **Positive** or **Negative** using transfer learning with DistilBERT. The system achieves **89.08% accuracy** on the test set.

The project uses the **IMDB movie review dataset** from Hugging Face, which contains 50,000 movie reviews labeled as positive or negative. The dataset is balanced during training to ensure fair model performance.

## Features

- Fine-tuned DistilBERT model on IMDB dataset
- FastAPI REST API for sentiment prediction
- Streamlit web interface for interactive analysis
- CPU-optimized for easy deployment

## Project Structure

```
customer_review_dsti_exam_2025/
├── project.ipynb          # Model training and evaluation notebook
├── run.py                 # Launches FastAPI and Streamlit apps
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── fastapi_app.py        # Generated FastAPI app (created on first run)
└── streamlit_app.py      # Generated Streamlit app (created on first run)
```

**Note**: The `fastapi_app.py` and `streamlit_app.py` files are automatically generated when you run `run.py` for the first time.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Internet connection (for downloading the model on first run)

### Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd customer_review_dsti_exam_2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install pandas numpy matplotlib seaborn scikit-learn datasets jupyter
```

**Note**: On the first run, the application will download the fine-tuned model from Hugging Face (~250MB). This may take a few minutes depending on your internet connection.

## Usage

### Run the Application

Simply run:
```bash
python run.py
```

This will:
- Start FastAPI server at `http://localhost:8000`
- Start Streamlit app at `http://localhost:8501`
- Automatically open the Streamlit interface

Press `Ctrl+C` to stop both servers.

### Train the Model

Open the Jupyter notebook:
```bash
jupyter notebook project.ipynb
```

Run all cells sequentially to:
- Load and preprocess the IMDB dataset
- Fine-tune the DistilBERT model (takes ~30-60 minutes on CPU)
- Evaluate model performance with metrics and visualizations
- Save the trained model locally

**Training Time**: Approximately 30-60 minutes on CPU for 2 epochs with 4,500 training samples.

## Model Details

- **Base Model**: `distilbert-base-uncased`
- **Task**: Binary classification (Positive/Negative)
- **Max Sequence Length**: 256 tokens (optimized for CPU)
- **Dataset**: IMDB movie reviews from Hugging Face
- **Training Samples**: 4,500 (2,250 positive + 2,250 negative)
- **Validation Samples**: 450 (10% split from training)
- **Test Samples**: 2,500 (1,250 positive + 1,250 negative)
- **Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear learning rate scheduler
- **Gradient Clipping**: Max norm 1.0

### Performance

- **Test Accuracy**: 89.08%
- **Test Loss**: 0.2936
- **Precision**: 0.89 (macro avg)
- **Recall**: 0.89 (macro avg)
- **F1-Score**: 0.89 (macro avg)

**Training Results**:
- Epoch 1: Train accuracy 81.23%, Validation accuracy 87.11%
- Epoch 2: Train accuracy 92.30%, Validation accuracy 88.00%

The fine-tuned model is available on Hugging Face: `akramxhs/distillbert_movie_reviews_sentiment`

## API Endpoints

### Health Check
```http
GET http://localhost:8000/
```

### Predict Sentiment
```http
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.9798,
  "text_preview": "This movie was absolutely fantastic!"
}
```

Interactive API documentation available at `http://localhost:8000/docs`

### Example Usage

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was amazing! Great acting and storyline."}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was terrible and boring."}'
```

## Technologies

- **PyTorch** - Deep learning framework for model training and inference
- **Transformers (Hugging Face)** - Pre-trained models and tokenizers
- **FastAPI** - Modern web framework for building REST APIs
- **Streamlit** - Rapid web app development for user interface
- **Uvicorn** - ASGI server for FastAPI
- **Pandas, NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Model evaluation metrics
- **Matplotlib, Seaborn** - Data visualization
- **Datasets** - Loading and managing datasets from Hugging Face

## Team

This project was developed by the following team members:

- **kawtar abidine** (DS - Data Scientist)
- **Nathalie Gálvez** (DS - Data Scientist)
- **HADJ SEYD Nazim Akram** (DS - Data Scientist)
- **Mohammed Ben slika** (DS - Data Scientist)
- **Yamina BAIT** (DE - Data Engineer)
- **K.G.L.G.Dayananda** (DE - Data Engineer)


Developed for the Deep Learning Project at DSTI (Data Science Tech Institute) - Exam 2025
