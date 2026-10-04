# Facial Feature Descriptor & Embedding Inference Engine

<p align="center">

### Deep Computer Vision • Facial Embeddings • Vector Similarity • Identity Inference

A deep-learning computer vision pipeline for **face detection, facial feature extraction, embedding-vector generation, similarity-based identity inference, and classroom attendance analytics**.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit&logoColor=white)
![ONNX](https://img.shields.io/badge/Models-ONNX-005CED?logo=onnx&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)

</p>

---
note🔥: Pretrained ONNX model weights are not included in the repository. Download the required YuNet and SFace model files separately and place them in the models/ directory.

## Overview

**Facial Feature Descriptor & Embedding Inference Engine** is a modular computer vision system that performs identity inference from facial images using pretrained deep-learning models.

The system follows a two-stage recognition pipeline:

1. **YuNet** — facial detection and landmark localization
2. **SFace** — aligned-face feature extraction and facial embedding generation

Instead of comparing images directly at the pixel level, the system transforms each detected face into a **deep feature embedding** and performs identity matching in the embedding space using **cosine similarity**.

The inferred identity is then mapped to a classroom attendance record.

```text
                    COMPUTER VISION INFERENCE PIPELINE

 ┌─────────────────┐
 │   Input Image   │
 └────────┬────────┘
          │
          ▼
 ┌────────────────────────┐
 │  YuNet Face Detection  │
 │  Bounding Box +        │
 │  Facial Landmarks      │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │ Face Alignment / Crop  │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │   SFace Feature Model   │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │ Facial Feature Vector   │
 │ / Embedding             │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │ Cosine Similarity       │
 │ Against Registered      │
 │ Embeddings              │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │ Identity Inference      │
 └──────────┬─────────────┘
            │
            ▼
 ┌────────────────────────┐
 │ Attendance Generation   │
 └────────────────────────┘
-----


##System Architecture
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
       ┌───────────────────┐                ┌────────────────────┐
       │ Student           │                │ Attendance          │
       │ Registration      │                │ Inference           │
       └─────────┬─────────┘                └─────────┬──────────┘
                 │                                    │
                 ▼                                    ▼
       ┌───────────────────┐                ┌────────────────────┐
       │ YuNet Detector    │                │ YuNet Detector     │
       └─────────┬─────────┘                └─────────┬──────────┘
                 │                                    │
                 ▼                                    ▼
       ┌───────────────────┐                ┌────────────────────┐
       │ Face Alignment    │                │ Multiple Face      │
       │ + Crop            │                │ Detection           │
       └─────────┬─────────┘                └─────────┬──────────┘
                 │                                    │
                 ▼                                    ▼
       ┌───────────────────┐                ┌────────────────────┐
       │ SFace Feature     │                │ SFace Feature      │
       │ Extraction        │                │ Extraction         │
       └─────────┬─────────┘                └─────────┬──────────┘
                 │                                    │
                 ▼                                    ▼
       ┌───────────────────┐                ┌────────────────────┐
       │ Reference         │                │ Query Embedding    │
       │ Embedding         │                └─────────┬──────────┘
       └─────────┬─────────┘                          │
                 │                                    │
                 └────────────────┬───────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ Embedding Similarity     │
                    │ / Identity Matching      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Identity Inference       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Attendance Record        │
                    └──────────────────────────┘
-----

Recognition Pipeline
1. Face Detection — YuNet

The first stage detects faces from the input image.

For every detected face, YuNet provides a facial bounding box and facial landmark information.
Input Image
     │
     ▼
┌─────────────────┐
│      YuNet      │
│ Face Detector   │
└────────┬────────┘
         │
         ▼
Face Bounding Box
+
Facial Landmarks
-----

##2. Face Alignment

The detected facial landmarks are passed to the SFace recognition pipeline for alignment and cropping.

Alignment helps normalize the facial region before feature extraction.

Detected Face
      │
      ▼
Facial Landmarks
      │
      ▼
Alignment
      │
      ▼
Normalized Face Crop

----
Deep Facial Feature Extraction — SFace

The aligned face is passed through the pretrained SFace recognition model.

SFace generates a learned feature representation of the face.

Aligned Face
      │
      ▼
┌───────────────┐
│     SFace     │
└───────┬───────┘
        │
        ▼
Deep Feature Vector
        │
        ▼
Facial Embedding

The resulting embedding captures discriminative facial characteristics in a numerical vector space.

---
                         Query Embedding
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Student A         Student B         Student C
        Embedding         Embedding         Embedding
             │                 │                 │
             ▼                 ▼                 ▼
        Similarity A      Similarity B      Similarity C
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                      Highest Similarity
                               │
                               ▼
                       Identity Candidate
                               │
                               ▼
                        Threshold Check
---

Student Registration

The registration pipeline creates the reference representation used during future inference.

Student ID
Name
College
Reference Image
      │
      ▼
Input Validation
      │
      ▼
Face Detection
      │
      ▼
Single-Face Validation
      │
      ▼
Face Alignment
      │
      ▼
SFace Embedding
      │
      ▼
Persist Metadata
+
Persist Embedding

----
Multi-Face Classroom Inference

The attendance pipeline supports classroom images containing multiple detected faces.

For each detected face:

Face #1 ──→ Embedding ──→ Registered Embedding Search
Face #2 ──→ Embedding ──→ Registered Embedding Search
Face #3 ──→ Embedding ──→ Registered Embedding Search
...
Face #N ──→ Embedding ──→ Registered Embedding Search

Each query face is evaluated independently against the registered identity set.

---
Technology Stack:

Layer	Technology
Programming Language	Python
Application Framework	Streamlit
Computer Vision	OpenCV
Face Detection	YuNet
Face Recognition / Feature Extraction	SFace
Model Runtime	OpenCV DNN / ONNX
Numerical Processing	NumPy
Data Processing	Pandas
Image Processing	OpenCV / Pillow
Attendance Storage	CSV
Spreadsheet Export	OpenPyXL
UI	Streamlit
Version Control	Git / GitHub

----
Project Structure:
facial-feature-descriptor-embedding-inference-engine/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── face_engine.py
│   └── attendance.py
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
└── data/
    ├── students.json
    ├── students/
    └── attendance/

----
Student Metadata:

Stored in:

data/students.json
Reference Images

Stored in:

data/students/
Facial Embeddings

Stored as NumPy arrays:

*.npy
Attendance

Stored locally as CSV and can be exported to Excel.

----
Application Modules
Dashboard

Provides an overview of the registered student database and attendance activity.

Student Registration

Creates a student's reference identity using:

Student ID
Name
College
Reference photograph / camera capture
Take Attendance

Processes a classroom image containing multiple faces and performs embedding-based identity inference.

Attendance History

Provides date-wise attendance records and supports Excel export.

Engineering Decisions
Why YuNet?

YuNet provides a lightweight face detection stage suitable for CPU-oriented inference and integrates directly with OpenCV's modern face APIs.

Why SFace?

SFace provides a pretrained deep face recognition pipeline that can generate discriminative facial features and perform similarity-based recognition.

Why Embeddings?

Embedding-based matching separates:

Image Acquisition
        ↓
Feature Extraction
        ↓
Identity Comparison

This makes the recognition layer independent of raw pixel-level image comparison.

Why Cosine Similarity?

Cosine similarity measures the angular relationship between embedding vectors and is commonly suitable for comparing feature vectors where direction in embedding space carries discriminative information.

Why Modular Architecture?

Separating the inference engine from persistence and UI allows the system to evolve toward:

REST APIs
Vector databases
Real-time inference
Cloud deployment
Scalable identity search

without rewriting the complete application.

---
Limitations:

This implementation is an educational / prototype-level biometric inference system.

Recognition quality can be affected by:

Illumination changes
Facial pose
Occlusion
Camera quality
Resolution
Motion blur
Facial expression
Detection quality
Reference-image quality

The similarity threshold is an operational parameter and has not been presented as a formally benchmarked accuracy value.

For production deployment, the threshold should be calibrated using a representative validation dataset containing genuine and impostor comparisons.

----
Scalability Considerations

The current implementation performs similarity comparisons against the registered embedding collection.

For a small classroom-sized identity set, this approach is straightforward and practical.

For large-scale deployments, the matching layer can be replaced with an approximate nearest-neighbor retrieval architecture:

                    Query Embedding
                           │
                           ▼
                 ┌───────────────────┐
                 │ Vector Index       │
                 │                   │
                 │ FAISS / Vector DB │
                 └─────────┬─────────┘
                           │
                           ▼
                    Top-K Candidates
                           │
                           ▼
                    Similarity Filter
                           │
                           ▼
                    Identity Inference

Potential future infrastructure:

FAISS
Vector databases
PostgreSQL + pgvector
REST API
GPU inference
Distributed storage

---
Deployment:
Streamlit Prototype
        ↓
FastAPI Inference Service
        ↓
PostgreSQL / Vector Store
        ↓
Containerized Deployment
        ↓
Cloud Infrastructure


---
Privacy & Security Considerations

Facial embeddings and facial images can constitute sensitive biometric information.

A production implementation should consider:

Explicit user consent
Secure storage
Encryption at rest
Encryption in transit
Access control
Data retention policies
Deletion mechanisms
Audit logging
Minimal data collection
Applicable biometric-data regulations

This repository is intended primarily for educational and experimental use.

---
License:

This project is intended for educational and experimental purposes.

If you plan to deploy or redistribute the pretrained models, review their respective model and dataset licensing terms separately.
----
Author

Pranav Sharma

Computer Science (AI/ML) Undergraduate
Interested in Machine Learning • Deep Learning • Generative AI • AI Engineering

GitHub: @Pranav123221
