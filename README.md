# AI Learning Assistant 🤖

AI Learning Assistant is an intelligent educational web application built with **Streamlit** that leverages **Natural Language Processing (NLP)** and **Deep Learning** to understand user emotions and provide interactive learning support. The application integrates **BiLSTM** and **BERT** models for emotion classification, enabling more personalized and engaging interactions.

An AI-powered Learning Assistant built using **Python**, **Streamlit**, **PyTorch**, **BiLSTM**, and **BERT** for emotion detection and intelligent learning assistance.

## Project Overview

The AI Learning Assistant is designed to enhance the learning experience by analyzing users' emotional states from text input and adapting interactions accordingly. It combines modern NLP techniques with deep learning models to create an intelligent educational assistant capable of understanding context, classifying emotions, and maintaining conversation history through an intuitive Streamlit interface.

## Features

- Interactive Streamlit Web Interface
- Emotion Detection using BiLSTM
- BERT-based Emotion Prediction
- Intelligent AI Learning Assistant
- Conversation History Storage
- Custom Tokenizer
- Vocabulary Builder
- Model Training Scripts
- Real-time User Interaction

---

## Project Structure

```
AI_LEARNING_ASSISTANT/
│
├── app.py
├── config.py
├── engine.py
├── model.py
├── predict.py
├── predict_bert.py
├── tokenizer.py
├── train.py
├── train_bert.py
├── vocab_builder.py
│
├── dataset.csv
├── data.csv
├── label_classes.npy
├── session_history.csv
│
├── bilstm_emotion_model/
│
├── vocab/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Streamlit
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Hugging Face Transformers
- BiLSTM
- BERT

---

## Installation

Clone the repository

```bash
git clone https://github.com/sivaadapa-ops/Emotion-Detection-and-Learning-Support-Engine/tree/main
```

Move into the project directory

```bash
cd AI-Learning-Assistant
```

Install dependencies

Ensure Streamlit is installed (included in requirements.txt):

```bash
pip install streamlit
```

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application

```bash
streamlit run app.py
```

## Training the BiLSTM Model

```bash
python train.py
```

---

## Training the BERT Model

```bash
python train_bert.py
```

---

## Predicting Emotion

Using BiLSTM

```bash
python predict.py
```

Using BERT

```bash
python predict_bert.py
```

---

## Dataset

The project uses CSV datasets for training and prediction.

- dataset.csv
- data.csv

---

## Model Files

The trained models are stored inside

```
bilstm_emotion_model/
```

Vocabulary files are stored inside

```
vocab/
```

---

## Future Improvements

- Voice Interaction
- AI Tutor
- Learning Progress Tracking
- Personalized Recommendations
- Dashboard
- Multi-language Support
- Better UI

---

## Author

Developed as a Machine Learning and NLP project using Python and PyTorch.
