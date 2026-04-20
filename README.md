# LELA60342_14332995

## Overview 
This repository utilises CUDA in the Computational Shared Facility to investigate whether the different types of vectorizers have an effect on accuracy. 
<br/> 
**Therefore, to compare this, two different models are trained and evaluated** 
- One that focuses on CountVectorizer: Counts the raw number of occurrences of each word in a document.
- One that focuses on TF-IDF: Weights term frequency against how rarely the word appears across all documents, downweighting common terms.
</br>
An example of different vectorisation methods is shown in the data.out file. As seen, in TF-IDF vectorisation, the numbers are displayed as a percentage, while CountBased only relies on 1 and 0. 

## Model Description   
**Two models, one with Count based vectorizer, and another with TFIDF vectorizer**
Both models share the same architecture: a single linear layer with sigmoid activation (`nn.Linear` → `torch.sigmoid`), trained with Binary Cross-Entropy Loss (`BCELoss`) and the Adam optimiser (lr=0.001) for 10 epochs with a batch size of 64, running on GPU via CUDA. The only difference is the input representation:
1. **Model 1 (TF-IDF)**: Input vectors are TF-IDF weighted.
2. **Model 2 (CountVectorizer)**: Input vectors are raw term counts.
<br/>
**Hypothesis** : Due to TFIDF vectorizer being more detailed than a simple count based method, I believe TFIDF will outperform the count based method

## Results

| Model | Vectoriser | Accuracy | AUC |
|---|---|---|---|
| Model 1 | TF-IDF | 0.8427 | 0.9262 |
| Model 2 | CountVectorizer | 0.8632 | 0.9285 |

**Bootstrap p-value (AUC Model 2 > Model 1): 0.065** (1000 samples)

## Result Discussion

Model 2 (CountVectorizer) outperformed Model 1 (TF-IDF) on both accuracy (+2.05%) and AUC (+0.0023). This proves my initial hypothesis wrong. The result could be attributable to TF-IDF's IDF weighting penalising high-frequency sentiment words (e.g. "great", "terrible") that appear commonly across reviews, thereby reducing the influence of the most discriminative features. On the other hand, raw counts preserve the full weight of these terms, making them more informative for the logistic regression classifier.

However, the bootstrap p-value of 0.065 does not reach a high enough statistical significance (p < 0.05). This means we cannot confidently conclude that Model 2 is a genuine improvement over Model 1 — the observed difference may be due to sampling variation. A larger dataset or more training epochs may be needed to produce a statistically significant result.  
