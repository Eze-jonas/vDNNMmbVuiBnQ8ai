#### AI-Powered Candidate Ranking System

Using Embeddings, LLM Prompt Engineering, and Vector Search

#### Overview

This project builds an end-to-end AI-driven candidate ranking system using classical NLP methods, large language models (LLMs), and retrieval-based techniques. It leverages talent sourcing and management datasets to predict candidate fitness, rank and re-rank candidates, ground LLM explanations, and fine-tune an LLM for recruitment-specific tasks.

## Data Availability
The datasets used in this project were provided by a talent sourcing and management company and are proprietary.
As a result, they are not included in this repository.

All notebooks, preprocessing steps, and modeling pipelines are provided to ensure reproducibility
given access to the same or a similar dataset.

#### Language: Python

#### Objectives

* Predict candidate fitness for specific roles

* Rank and re-rank candidates based on relevance

* Ground LLM-generated explanations in retrieved data

* Fine-tune an LLM to specialize in recruitment and candidate evaluation

#### Key Components
1. **Exploratory Data Analysis (EDA)**

Candidate profile datasets containing keywords such as “Aspiring human resources” and “seeking human resources” were cleaned and preprocessed. All preprocessing and analysis steps are documented in the accompanying notebooks.

2. **Fitness Prediction and Ranking**

* Embedding-based ranking:
Candidates were ranked using word- and sentence-level embeddings (TF-IDF, GloVe, FastText, SBERT) with cosine similarity.

* LLM-based ranking:
Candidates were also ranked using LLMs (LLaMA, Qwen, Gemini API) via prompt-based methods.
Greedy decoding was applied to reduce hallucinations and ensure deterministic outputs.

3. **Re-ranking and Grounded Explanations (RAG)**

A Retrieval-Augmented Generation (RAG) system was developed using WSL and a Linux virtual environment compatible with FAISS.

* Retriever: Sentence embeddings + FAISS vector database to retrieve relevant candidate profiles

* Re-ranker: Cross-encoder to re-rank retrieved candidates using query–profile semantic matching

* Generator: LLaMA with a greedy decoder and structured prompts to generate explanations grounded strictly in retrieved and re-ranked candidates

This ensured transparent, evidence-based candidate recommendations.

4. **LLM Fine-Tuning for Recruitment**

A candidate dataset with screening scores was used to fine-tune LLaMA using Low-Rank Adaptation (LoRA) in Google Colab.
The fine-tuned model was evaluated via inference to assess its specialization in recruitment and candidate evaluation tasks.

#### Results

* Successfully predicted candidate fitness and ranked candidates using embedding-based similarity and LLM reasoning

* Improved ranking quality through cross-encoder re-ranking

* Grounded LLM explanations using a RAG framework

* Specialized LLaMA for recruitment tasks via LoRA fine-tuning

#### Conclusions/Recommendations

* Adopt multi-stage candidate ranking (retrieval → re-ranking → grounded LLM explanations)

* Use domain-specialized fine-tuned LLMs for recruitment

* Keep humans in the loop for final decisions

* Track evaluation metrics for fairness and effectiveness

* Source larger and more diverse candidate datasets to improve model generalization, ranking stability, and fairness evaluation.

#### Notebooks & Project Structure
* Project_Summary
* EDA_For_Fine_Tunning.ipynb
* EDA_For_Fine_Ranking.ipynb
* Ranking_with_FastText.ipynb
* Ranking_with_GloVe.ipynb
* Ranking_with_Gemini_API.ipynb
* Ranking_with_TF_IDF.ipynb
* Ranking_with_SBERT.ipynb
* Rag_System
* LlaMA_Fine_Tunning.ipynb
* Fine_Tuned_LlaMA_Evaluation.ipynb
* README

#### Keywords

**NLP · Candidate Ranking · Embeddings · FAISS · RAG · LLMs · LoRA · Recruitment AI · Explainable AI**


```python

```
