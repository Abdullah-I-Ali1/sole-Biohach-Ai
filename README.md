
## Overview

This project aims to analyze the genomic signatures of SARS-CoV-2 using machine learning (ML) and deep learning (DL) techniques in an alignment-free approach.

## Problem Statement

Traditional alignment-based phylogenetic methods require significant computational resources. This project provides an efficient alternative by extracting genomic signatures and applying ML and DL models to classify SARS-CoV-2 lineages.

## Steps Taken 
# Data Processing

Read genomic sequence data from input files.

Clean and preprocess the data to remove invalid values.

Feature Extraction

Compute dinucleotide and trinucleotide frequencies using Python.

Store the extracted features in a suitable format for model training.

Model Development

Implemented multiple machine learning models:

Random Forest

Support Vector Machine (SVM)

Logistic Regression


Developed a deep learning model using Keras/TensorFlow.

Split the data into training and testing sets for evaluation.

Evaluation and Analysis

Calculated accuracy and other performance metrics to compare model effectiveness.

Analyzed results to identify SARS-CoV-2 lineages with high precision.

Implementation
The entire workflow was implemented using 100% Python, utilizing libraries such as scikit-learn, TensorFlow, Pandas, and NumPy.
