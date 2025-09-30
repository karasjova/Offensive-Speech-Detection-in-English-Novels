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

- **Original texts**: *The Catcher in the Rye* (`dataset/Salinger - The Catcher in the Rye -English.txt`), *Last Exit to Brooklyn* (`dataset/last-exit-to-brooklyn.txt`)  
- **HurtLex lexicon**: curated lexicon of offensive words (`dataset/hurtlex_filtered.xlsx`)
- **Final lexicon**: expanded lexicon of offensive words (`dataset/final_lexicon.csv`)
- **Labelled sentences**: manually annotated sentences for model fine-tuning (`dataset/new_salinger_labelled.csv`)  

