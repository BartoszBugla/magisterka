# ABSA Sentiment Analysis Dashboard

Aspect-Based Sentiment Analysis (ABSA) for location reviews with interactive map visualization.

## Features

- **BERT-based ABSA Model**: Fine-tuned BERT for multi-label aspect sentiment classification
- **7 Aspects**: Tourism, Infrastructure, Health, Safety, Culture, Heritage, Other
- **3 Sentiments**: Positive, Neutral, Negative
- **Interactive Map**: Visualize sentiment geographically with heatmaps and 3D views
- **Live Prediction**: Analyze custom reviews in real-time

## Quick Start

```bash
# Install dependencies
uv pip install streamlit pydeck torch transformers pandas numpy

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Streamlit dashboard
├── experiments/
│   ├── absa-model.ipynb           # Model training notebook
│   └── absa_model/                # Saved model weights
├── datasets/
│   ├── cleaned_reviews.csv        # Labeled reviews with coordinates
│   └── absa_training_dataset.csv  # 1000-row training dataset
└── README.md
```

## Dashboard Features

### Map Visualization
- **Points**: Color-coded markers showing sentiment per location
- **Heatmap**: Density visualization of sentiment intensity
- **3D Hexagons**: Elevated hexagonal bins showing sentiment volume
- **Combined**: All visualization types together

### Controls
- Select aspect to visualize (tourism, safety, etc.)
- Filter by sentiment (positive/neutral/negative)
- Adjust zoom, point size, and 3D pitch
- Search reviews by keyword

### Live Prediction
- Enter any review text
- Get real-time aspect-sentiment predictions
- View confidence scores and probability breakdown
