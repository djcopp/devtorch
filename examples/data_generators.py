"""
Synthetic Data Generators for DevTorch Examples

This module contains functions to generate synthetic datasets for various example types.
All fake data construction is centralized here to keep example files clean and focused.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Any
from sklearn.datasets import fetch_california_housing, make_classification


def generate_text_classification_data(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Generate synthetic text classification data for sentiment analysis."""
    
    np.random.seed(42)
    
    # Text templates for different sentiments
    positive_templates = [
        "This {product} is absolutely {adjective}! I {verb} it so much.",
        "Amazing {product} with {adjective} {feature}. Highly recommend!",
        "Love this {product}! The {feature} is {adjective} and works perfectly.",
        "Outstanding {product}! {adjective} quality and {adjective} performance."
    ]
    
    negative_templates = [
        "This {product} is {adjective}. The {feature} {verb} constantly.",
        "Terrible {product}! {adjective} quality and poor {feature}.",
        "Waste of money. The {product} is {adjective} and {feature} is broken.",
        "Awful {product}. {adjective} design and {feature} doesn't work."
    ]
    
    neutral_templates = [
        "The {product} is okay. {adjective} {feature} but nothing special.",
        "Average {product}. The {feature} works but could be {adjective}.",
        "Decent {product} with {adjective} {feature}. Not bad, not great.",
        "Standard {product}. {feature} is {adjective} for the price."
    ]
    
    products = ["phone", "laptop", "headphones", "camera", "tablet", "speaker"]
    positive_adjectives = ["fantastic", "excellent", "amazing", "wonderful", "brilliant"]
    negative_adjectives = ["terrible", "awful", "horrible", "disappointing", "useless"]
    neutral_adjectives = ["decent", "acceptable", "adequate", "reasonable", "fair"]
    features = ["battery", "screen", "sound quality", "design", "performance", "interface"]
    positive_verbs = ["love", "enjoy", "adore", "appreciate"]
    negative_verbs = ["fails", "breaks", "crashes", "malfunctions"]
    
    texts = []
    labels = []
    
    # Generate samples for each class
    samples_per_class = n_samples // 3
    
    # Positive samples (label 2)
    for _ in range(samples_per_class):
        template = np.random.choice(positive_templates)
        text = template.format(
            product=np.random.choice(products),
            adjective=np.random.choice(positive_adjectives),
            feature=np.random.choice(features),
            verb=np.random.choice(positive_verbs)
        )
        texts.append(text)
        labels.append(2)
    
    # Negative samples (label 0)
    for _ in range(samples_per_class):
        template = np.random.choice(negative_templates)
        text = template.format(
            product=np.random.choice(products),
            adjective=np.random.choice(negative_adjectives),
            feature=np.random.choice(features),
            verb=np.random.choice(negative_verbs)
        )
        texts.append(text)
        labels.append(0)
    
    # Neutral samples (label 1)
    for _ in range(n_samples - 2 * samples_per_class):
        template = np.random.choice(neutral_templates)
        text = template.format(
            product=np.random.choice(products),
            adjective=np.random.choice(neutral_adjectives),
            feature=np.random.choice(features)
        )
        texts.append(text)
        labels.append(1)
    
    return texts, labels


def generate_time_series_data(n_timesteps: int = 2000, n_features: int = 3) -> np.ndarray:
    """Generate synthetic time series data with trend and seasonality."""
    
    np.random.seed(42)
    t = np.linspace(0, 10, n_timesteps)
    data = np.zeros((n_timesteps, n_features))
    
    # Feature 0: Trend + seasonality + noise
    data[:, 0] = (
        0.1 * t +  # Linear trend
        2 * np.sin(2 * np.pi * t / 50) +  # Seasonality
        0.5 * np.random.randn(n_timesteps)  # Noise
    )
    
    # Feature 1: Different seasonal pattern
    data[:, 1] = (
        1.5 * np.cos(2 * np.pi * t / 30) +
        0.3 * np.random.randn(n_timesteps)
    )
    
    # Feature 2: More complex pattern
    data[:, 2] = (
        np.sin(t) * np.cos(0.5 * t) +
        0.2 * np.random.randn(n_timesteps)
    )
    
    return data


def generate_tabular_regression_data(n_samples: int = 5000, n_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic tabular data for regression with heteroscedastic noise."""
    
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with complex relationships and input-dependent noise
    y = (
        2.0 * X[:, 0] +
        0.5 * X[:, 1] +
        3.0 * np.sin(X[:, 2]) +
        1.5 * X[:, 3] * X[:, 4] +  # Interaction term
        np.random.normal(0, 0.2 + 0.5 * np.abs(X[:, 0]))  # Heteroscedastic noise
    )
    
    return X, y


def get_california_housing_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the California housing dataset."""
    
    housing = fetch_california_housing()
    return housing.data, housing.target, list(housing.feature_names)


def generate_multimodal_classification_data(n_samples: int = 500) -> Tuple[List[str], List[int]]:
    """Generate synthetic multimodal data (text descriptions and labels)."""
    
    np.random.seed(42)
    
    # Text descriptions for different categories
    templates = {
        0: [  # Vehicles
            "A {color} {vehicle} driving on the {location}",
            "Fast {color} {vehicle} racing on the track",
            "Vintage {vehicle} from the {era} era",
            "Modern {color} {vehicle} with sleek design"
        ],
        1: [  # Buildings
            "Modern {type} with {material} construction",
            "Contemporary {type} with {feature} design", 
            "{type} with {feature} windows and {material} facade",
            "Architectural {type} featuring {material} and glass"
        ],
        2: [  # Food
            "Fresh {color} {food} with {ingredient}",
            "Delicious {food} dish with {ingredient} and herbs",
            "Gourmet {food} served with {ingredient}",
            "Traditional {food} prepared with {ingredient}"
        ],
        3: [  # Nature
            "Colorful {plant} blooming in the {season}",
            "Bright {color} {plant} in the {location}",
            "Beautiful {plant} growing in {season}",
            "Wild {plant} found in the {location}"
        ]
    }
    
    vocab = {
        'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange'],
        'vehicle': ['car', 'motorcycle', 'truck', 'bicycle', 'bus'],
        'location': ['highway', 'street', 'mountain road', 'city center'],
        'era': ['1960s', '1970s', '1980s', 'classic'],
        'type': ['building', 'skyscraper', 'house', 'tower', 'apartment'],
        'material': ['glass', 'steel', 'concrete', 'brick', 'wood'],
        'feature': ['geometric', 'curved', 'angular', 'reflective'],
        'food': ['salad', 'pasta', 'pizza', 'soup', 'sandwich'],
        'ingredient': ['tomatoes', 'cheese', 'herbs', 'vegetables', 'sauce'],
        'plant': ['flowers', 'trees', 'grass', 'bushes', 'sunflowers'],
        'season': ['spring', 'summer', 'autumn', 'winter']
    }
    
    texts = []
    labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, 4)
        template = np.random.choice(templates[label])
        
        # Fill template with random vocabulary
        text = template
        for key, values in vocab.items():
            if f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', np.random.choice(values))
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def generate_audio_classification_data(n_samples: int = 500) -> Tuple[List[str], List[int]]:
    """Generate synthetic audio file paths and labels."""
    
    np.random.seed(42)
    
    # Generate dummy audio file paths
    audio_files = [f"dummy_audio_{i:04d}.wav" for i in range(n_samples)]
    
    # Generate labels for different audio classes
    # Classes: 0=music, 1=speech, 2=environmental, 3=silence
    labels = np.random.randint(0, 4, n_samples).tolist()
    
    return audio_files, labels


def generate_synthetic_audio_signal(duration: float = 3.0, sample_rate: int = 22050) -> np.ndarray:
    """Generate a synthetic audio signal for demonstration."""
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Combine multiple frequency components
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note  
        0.2 * np.sin(2 * np.pi * 220 * t) +  # A3 note
        0.1 * np.random.randn(len(t))        # Noise
    )
    
    return audio


# Constants for commonly used class names
TEXT_SENTIMENT_CLASSES = ['Negative', 'Neutral', 'Positive']
MULTIMODAL_CLASSES = ['Vehicles', 'Buildings', 'Food', 'Nature']
AUDIO_CLASSES = ['Music', 'Speech', 'Environmental', 'Silence'] 