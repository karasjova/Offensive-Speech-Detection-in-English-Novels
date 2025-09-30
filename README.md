# Offensive-Speech-Detection-in-English-Novels

This repository contains a complete pipeline for detecting offensive language in English literary texts. The code combines lexicon-based methods with modern transformer-based models to identify and classify offensive sentences.

## Overview

This project implements a hybrid approach for offensive speech detection in English novels:

- **Lexicon-based filtering** using HurtLex.  
- **Word embedding similarity** to expand lexicons and find additional candidate offensive words.  
- **Transformer-based models** (HateBERT, HateXplain) for classification of filtered sentences. 

The pipeline allows for reproducible experiments and can be adapted to other literary texts or languages with suitable lexicons and fine-tuning.

## Dataset

The project uses:

- **Original texts**: *The Catcher in the Rye*, *Last Exit to Brooklyn*
- **HurtLex lexicon**: curated lexicon of offensive words (`dataset/hurtlex_filtered.xlsx`)
- **Final lexicon**: expanded lexicon of offensive words (`dataset/final_lexicon.csv`)
- **Annotated dataset**: manually annotated sentences for model fine-tuning (`dataset/new_salinger_labelled.csv`)

## Installation

```bash
pip install -r requirements.txt
```

## Notebooks Overview

## Usage

Run all cells in Jupyter Notebook, Kaggle, or Colab.

## 1. Final_lexicon_creation.ipynb
   
   The script creates a lexicon based on the initial Hurtlex lexicon, which is further expanded by using cosine similarity with tokens extracted from the processed literary text.

## 2. CM_training.ipynb

  The script performs training of machine learning models (Random Forest, SVM, Naive Bayes) with 80/20 train-test split and 5-fold cross-validation on annotated dataset.

 ## 3. PLM_training.ipynb

  The script performs training of pretrained language models (HateBert and HateXplain) with 80/20 train-test split and 5-fold cross-validation on annotated dataset.

## 4. Offensive_Speech_Detection_Pipeline.ipynb
  The script implements the complete pipeline for offensive speech detection, including text preprocessing, lexicon expanding, lexicon-based sentece filtering, and classification using pretrained language models.
  
