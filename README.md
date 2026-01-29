# News Recommendation System using Liquid Time-Constant Networks (LTC)

This project implements a **news recommendation system** using **Liquid Time-Constant (LTC) networks** to model user preferences.  
The system captures both **short-term** and **long-term** user interests from interaction data and supports **cold-start scenarios** with limited user history.

## 📁 Project Structure

NewsRecommenderSystem/  
├── data/ # MIND dataset  
├── preprocessing/                 `# Preprocessing pipelines`  
│ ├── dataset_ingestion.py  
│ ├── sequence_builder.py  
│ ├── short_term_preprocessing.py  
│ ├── long_term_preprocessing.py  
│ └── run_preprocessing_pipeline.py  
├── models/                        `# models`  
│ ├── embeddings.py  
│ ├── ltc_encoder.py  
│ ├── short_term.py  
│ └── long_term.py  
├── run_pipeline.py                `# End-to-end pipeline runner`   
└── README.md  

## Requirements

- Python **3.8 or higher**
- PyTorch
- ncps (Neural Circuit Policies)
- numpy
- tqdm

## Installation

Install the required dependencies using:

```bash
pip install torch ncps numpy tqdm
```

## How to Run

Ensure the MIND dataset is placed inside the data/ directory.
Run the complete pipeline:
```bash
python -m run_pipeline
```
This will:  
Construct user interaction sequences (once)  
Perform short-term and long-term preprocessing  
Generate user representations using LTC networks  
Successful execution prints the generated embedding shapes for verification  



