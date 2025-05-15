# Multimodal Emotion Analyzer with Video, Text, and Voice

A comprehensive SaaS platform for analyzing emotions and sentiments in videos using a multimodal deep learning approach. This project combines video frame analysis, audio processing, and text transcription to provide accurate emotion and sentiment detection.



## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [SaaS Platform](#saas-platform)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Deploying the Model](#deploying-the-model)
- [API Usage](#api-usage)
- [Technologies Used](#technologies-used)
- [License](#license)

## 🔍 Overview

This project is a complete end-to-end solution for video sentiment analysis. It consists of two main components:

1. **AI Model Component**: A multimodal deep learning model that processes video, audio, and text to detect emotions and sentiments.
2. **SaaS Web Application**: A web platform that allows users to upload videos, analyze them using the trained model, and view detailed results.

The system extracts frames from videos, transcribes speech, and analyzes audio patterns to provide comprehensive emotional analysis with high accuracy.

## ✨ Features

### ML Model Features
- 🎥 Video frame extraction and analysis
- 🎙️ Audio feature extraction using mel spectrograms
- 📝 Text embedding with BERT
- 🔗 Multimodal fusion of video, audio, and text features
- 📊 Emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise)
- 💭 Sentiment classification (positive, negative, neutral)
- 📈 TensorBoard logging for model training visualization
- 🚀 Model training and evaluation pipeline

### SaaS Platform Features
- 🔐 User authentication with Auth.js
- 🔑 API key management for secure access
- 📊 Usage quota tracking for API requests
- 🚀 Video upload and processing
- 📈 Real-time analysis results with confidence scores
- 🎨 Modern UI with Tailwind CSS
- 📱 Responsive design for all devices
- 📄 API documentation and code examples

## 🧠 Model Architecture
![Alt text](/Model_Working.png)

The diagram above illustrates our multimodal approach to emotion and sentiment analysis. The architecture consists of three parallel encoding pathways (video, text, audio) that are later fused to make predictions.

### Encoding Pathways

#### 1. Video Encoder

- **Input**: Extracts 30 frames from each video clip
- **Processing**: Uses a 3D ResNet (R3D-18) pretrained on action recognition
- **Feature Extraction**: Processes frames through convolutional layers to capture spatial and temporal patterns
- **Output**: Produces a 128-dimensional feature vector representing visual emotional cues


#### 2. Text Encoder

- **Input**: Transcribed speech from the video
- **Processing**: Uses BERT (bert-base-uncased) for contextual word embeddings
- **Feature Extraction**: The [CLS] token embedding captures sentence-level semantics
- **Output**: Projects to a 128-dimensional feature vector representing linguistic emotional cues


#### 3. Audio Encoder

- **Input**: Mel spectrograms extracted from the audio track
- **Processing**: Multi-stage convolutional network with batch normalization
- **Feature Extraction**:

- Lower-level features: Captures tone, pitch, and rhythm
- Higher-level features: Identifies patterns like stress and intonation



- **Output**: Projects to a 128-dimensional feature vector representing acoustic emotional cues


### Fusion and Classification

- **Feature Fusion**: Concatenates the three 128-dimensional vectors into a 384-dimensional multimodal representation
- **Fusion Layer**: Processes the concatenated features through a fully connected layer with batch normalization (256 neurons)
- **Classification Heads**:

- **Emotion Classifier**: Predicts 7 emotion classes (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Sentiment Classifier**: Predicts 3 sentiment classes (positive, negative, neutral)





### Training Components

As shown in the bottom part of the diagram:

- **Optimizer**: Adam optimizer with different learning rates for each component
- **Loss Function**: Cross-entropy loss with label smoothing for both emotion and sentiment tasks
- **Learning Rate Scheduler**: ReduceLROnPlateau that reduces learning rate when validation metrics plateau


This architecture effectively combines visual, linguistic, and acoustic information to provide a comprehensive analysis of emotions and sentiments in video content.

### Training Details

- Uses weighted cross-entropy loss with label smoothing
- Adam optimizer with different learning rates for each component
- ReduceLROnPlateau scheduler for adaptive learning rate
- Trained on the MELD dataset (Multimodal EmotionLines Dataset)
- Evaluation metrics: precision, accuracy for both emotion and sentiment


## 💻 SaaS Platform

The SaaS platform provides a user-friendly interface for accessing the video sentiment analysis capabilities:

### Authentication System

- Email/password registration and login
- Secure password hashing with bcrypt
- JWT-based session management with Auth.js


### API Key Management

- Automatic API key generation for each user
- Secure key storage and validation
- Usage tracking and quota enforcement


### Video Processing Pipeline

1. User uploads a video through the web interface or API
2. Video is stored securely in cloud storage
3. System generates a signed URL for the video
4. Model processes the video, extracting frames, audio, and text
5. Results are returned with detailed emotion and sentiment analysis


### Dashboard Features

- Real-time analysis results visualization
- Emotion and sentiment confidence scores
- Utterance-level breakdown of emotions and sentiments
- API usage statistics and quota monitoring
- Code examples for API integration


## 🚀 Setup and Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- AWS account (for model deployment)
- PostgreSQL or SQLite database


### Installation Steps

1. Clone the repository:


```shellscript
git clone https://github.com/whojayy/multimodal-emotion-analyzer-with-aid-of-Video-Text-and-Voice.git
cd multimodal-emotion-analyzer-with-aid-of-Video-Text-and-Voice
```

2. Install dependencies for the SaaS platform:


```shellscript
npm install
```

3. Set up environment variables:


```plaintext
DATABASE_URL="your-database-url"
AUTH_SECRET="your-auth-secret"
AWS_REGION="your-aws-region"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
AWS_INFERENCE_BUCKET="your-bucket-name"
AWS_ENDPOINT_NAME="your-endpoint-name"
```

4. Initialize the database:


```shellscript
npx prisma generate
npx prisma db push
```

5. Install Python dependencies for the model (if working with the model locally):


```shellscript
cd ai-video-sentiment-model/training
pip install -r requirements.txt
```

6. Start the development server:


```shellscript
npm run dev
```

## 🧪 Training the Model

### Dataset Preparation

The model is trained on the MELD dataset (Multimodal EmotionLines Dataset), which contains:

- 13,000+ utterances from Friends TV series
- 7 emotion labels and 3 sentiment labels
- Video, audio, and text for each utterance


1. Download the MELD dataset from the official source
2. Extract and place it in the `ai-video-sentiment-model/dataset` directory
3. Preprocess the dataset using the provided scripts


### Training Process

1. Configure training parameters in `train.py`
2. Start the training process:


```shellscript
cd ai-video-sentiment-model
python training/train.py
```

3. Monitor training progress with TensorBoard:


```shellscript
tensorboard --logdir runs/
```

4. The best model will be saved based on validation loss


### Training on AWS SageMaker

For large-scale training, the model can be trained on AWS SageMaker:

1. Configure SageMaker settings in `train_sagemaker.py`
2. Upload the dataset to an S3 bucket
3. Run the SageMaker training job:


```shellscript
python train_sagemaker.py
```

## 📊 Deploying the Model

### Local Deployment

For testing and development, you can run inference locally:

```shellscript
cd ai-video-sentiment-model/deployment
python inference.py --video_path path/to/video.mp4
```

### Cloud Deployment

For production use, deploy the model as an endpoint:

1. Package the model artifacts
2. Configure deployment settings in `deploy_endpoint.py`
3. Deploy the model:


```shellscript
cd ai-video-sentiment-model/deployment
python deploy_endpoint.py
```

4. The endpoint will be available for inference through the SaaS platform


## 🔌 API Usage

### Authentication

All API requests require an API key for authentication:

```shellscript
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fileType": ".mp4"}' \
  /api/upload-url
```

### Video Analysis

The API follows a three-step process:

1. Get a signed upload URL:


```shellscript
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fileType": ".mp4"}' \
  /api/upload-url
```

2. Upload the video to the provided URL:


```shellscript
curl -X PUT \
  -H "Content-Type: video/mp4" \
  --data-binary @video.mp4 \
  "SIGNED_URL"
```

3. Analyze the video:


```shellscript
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "FILE_KEY"}' \
  /api/sentiment-inference
```

### Response Format

The API returns a detailed analysis of the video:

```json
{
  "analysis": {
    "utterances": [
      {
        "start_time": 0.5,
        "end_time": 3.2,
        "text": "I can't believe this happened!",
        "emotions": [
          {"label": "surprise", "confidence": 0.85},
          {"label": "joy", "confidence": 0.10},
          {"label": "neutral", "confidence": 0.05}
        ],
        "sentiments": [
          {"label": "positive", "confidence": 0.75},
          {"label": "neutral", "confidence": 0.20},
          {"label": "negative", "confidence": 0.05}
        ]
      }
    ]
  }
}
```

## 🛠️ Technologies Used

### Machine Learning

- PyTorch - Deep learning framework
- torchvision - Computer vision library
- torchaudio - Audio processing library
- transformers - NLP models including BERT
- OpenAI Whisper - Speech recognition
- scikit-learn - Evaluation metrics
- TensorBoard - Training visualization


### Web Development

- Next.js - React framework
- TypeScript - Type-safe JavaScript
- Tailwind CSS - Utility-first CSS framework
- Prisma - Type-safe ORM
- Auth.js - Authentication library
- React Hook Form - Form validation
- Zod - Schema validation


### Cloud Infrastructure

- AWS S3 - Video storage
- AWS SageMaker - Model training and deployment
- AWS IAM - Access management




---

## Acknowledgements

- The MELD dataset creators for providing the multimodal emotion dataset
- The PyTorch team for the deep learning framework
- The Next.js team for the React framework