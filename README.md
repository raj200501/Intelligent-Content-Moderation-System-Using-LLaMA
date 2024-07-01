# Intelligent Content Moderation System Using LLaMA

This repository contains the code for an intelligent content moderation system that uses Meta's LLaMA for natural language understanding. The system detects and filters inappropriate content in real-time, leveraging advanced NLP techniques and machine learning models.

## Features

- Content Classification (using Meta AI tools)
- Sentiment Analysis
- Toxic Comment Detection
- Real-time Moderation
- Dockerized Deployment

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Docker Compose

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/intelligent-content-moderation.git
    cd intelligent-content-moderation
    ```

2. Build and run the Docker containers:
    ```bash
    docker-compose up --build
    ```

3. Load and preprocess data:
    ```bash
    python data/load_data.py
    python data/preprocess_data.py
    ```

4. Train and evaluate models:
    ```bash
    python models/train_model.py
    python models/evaluate_model.py
    ```

5. Run the moderation service:
    ```bash
    python backend/moderation_service/moderation.py
    ```

## License

This project is licensed under the MIT License.
