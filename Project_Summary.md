#### Project Summary/Conclusions.
#### Language: Python.
#### Title: AI-Powered Candidate Ranking System Using Embeddings, LLM Prompt Engineering and vector search

This project aimed to leverage a talent sourcing and management company dataset to build a candidate ranking system. Specifically:

1. **Using the candidate profile dataset with keywords (“Aspiring human resources” or “seeking human resources”) to:**

* Predict candidate fitness for specific roles.

* Rank candidates based on predicted fitness.

* Re-rank candidates and ground LLM explanations in retrieved data

2. **Using the candidate profile dataset with screening scores to:**

* Fine-tune LLMs to specialize in recruitment and candidate evaluation.

#### Exploratory Data Analysis (EDA)

The candidate profile dataset with keywords (“Aspiring human resources” or “seeking human resources" has 105 rows and 5 clolumns. It was cleaned and preprocessed; details are documented in the accompanying notebooks.

#### Fitness Prediction and Ranking

* First, the preprocessed dataset was used to predict candidate fitness. Candidates were then ranked using word-level and sentence-level embedding models (TF-IDF, GloVe, FastText, and SBERT) based on cosine similarity.

* Second, candidates were also ranked using LLMs (LLaMA, Qwen, and Gemini API) via a prompt-based approach, with greedy decoding applied to reduce hallucinations and ensure deterministic outputs.

### Re-ranking Candidates and Grounding LLM Explanations in Retrieved Data

This was implemented using the Windows Subsystem for Linux (WSL) and a Linux virtual environment, which are compatible with the FAISS library.

To re-rank candidates and ground LLM explanations in retrieved data, a retriever-augmented generation (RAG) system was developed.
* First, the retrieval component was built using a sentence embedder and a vector database (FAISS). This component predicted candidate fitness, ranked candidates, and retrieved the most relevant profiles.

* Second, a re-ranking component was built and integrated into the retrieval system using a cross-encoder. This component re-ranked the retrieved candidates by jointly encoding candidate profile–query pairs, ensuring that the final ranking was based on deep semantic relevance rather than vector similarity alone.

* Finally, the generation component was built and integrated with the retrieval and re-ranking system using an LLM (LLaMA), greedy decoder and a carefully designed prompt. This component generated explanations using only the retrieved and re-ranked candidate profiles, explaining why one candidate should be preferred over another, thereby ensuring that the LLM’s reasoning was grounded in retrieved evidence.

#### Fine-tunning LLM(LLaMA) to specialize in recruitment and candidate evaluation.

The candidate profile dataset with screening scores has 1285 rows and 4 columns It was cleaned and preprocessed; details are documented in the accompanying notebooks. The processed dataset was then used to fine-tune LLaMA using the Low-Rank Adaptation (LoRA) method in Google Colab. The fine-tuned model was subsequently evaluated through inference to assess its specialization in recruitment and candidate evaluation tasks.

#### Conclusions

This project successfully predicted candidate fitness and ranked candidates using cosine similarity–based methods, prompt-engineered large language models, and vector search. It further improved ranking quality by re-ranking candidates and grounding LLM explanations through the development of a Retrieval-Augmented Generation (RAG) system. Finally, the project specialized a LLaMA model for recruitment and candidate evaluation by fine-tuning it using the Low-Rank Adaptation (LoRA) technique.

#### Recommendations

* Adopt multi-stage candidate ranking (retrieval → re-ranking → grounded LLM explanations)

* Use domain-specialized fine-tuned LLMs for recruitment

* Keep humans in the loop for final decisions

* Track evaluation metrics for fairness and effectiveness

* Source larger and more diverse candidate datasets to improve model generalization, ranking stability, and fairness evaluation.


```python

```
