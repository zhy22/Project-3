
**Project 3** — An end-to-end NLP solution to automatically rank and score job candidates for Human Resources (HR) roles based on their profiles (job title, location, and network connections).

This project demonstrates modern text processing techniques, semantic similarity, prompt engineering with large language models (LLMs), and parameter-efficient fine-tuning to predict a **fit score** (0–1) indicating how well a candidate matches HR-related opportunities.

## Project Overview

The goal is to build an intelligent candidate screening tool that mimics HR recruiter judgment using only lightweight profile data:
- **Input features**: Job title, location, number of connections (e.g., LinkedIn-style).
- **Output**: A fit score [0, 1] — higher for candidates explicitly aspiring to or working in HR roles.
- **Dataset**: ~100 anonymized candidate profiles (likely synthetic or public LinkedIn-derived data).
- **Key challenge**: Accurately interpret varied job titles and boost relevant candidates without labeled training data initially.

The project evolves in two parts:
- **Part I**: Explores multiple embedding-based similarity methods + zero-shot LLM prompting.
- **Part II**: Fine-tunes an LLM with PEFT/LoRA + SFT for improved, calibrated predictions.

Technologies used:
- Embeddings: TF-IDF, Word2Vec, GloVe, FastText, BERT (sentence-transformers)
- LLMs: Llama-3.1-8B-Instruct (quantized), Grok API, attempted Qwen
- Fine-tuning: Hugging Face Transformers, PEFT (LoRA), TRL (SFTTrainer), bitsandbytes

## Project Structure

## Step-by-Step Project Flow

### Part I – Baseline & Prompting Approaches

1. **Data Loading & Preprocessing**  
   Load candidate DataFrame (id, job_title, location, connection).  
   Clean text (lowercase, remove special chars).  
   Parse connections (e.g., "500+" → 500) and normalize logarithmically to [0,1].

2. **Keyword Boosting**  
   Binary feature: 1 if title contains phrases like "aspiring human resources" or "seeking human resources".

3. **Semantic Similarity Modeling** (multiple variants compared):  
   - **TF-IDF + Cosine**: Vectorize text, compare to HR prototype phrase.  
   - **Word2Vec**: Train custom embeddings, average vectors, cosine sim.  
   - **Pre-trained GloVe / FastText**: Load embeddings, average, cosine.  
   - **BERT (MiniLM)**: Contextual sentence embeddings via sentence-transformers.  
   → Compute similarity to HR prototype ("human resources hr recruiter ...").

4. **Fit Score Fusion**  
   Weighted combination:  
   `fit = 0.65 × keyword + 0.30 × similarity + 0.05 × normalized connections`

5. **LLM Prompting (Zero-shot)**  
   Use Llama-3.1-8B-Instruct (4-bit quantized) and Grok API.  
   Structured system prompt → JSON output `{"fit": <score>}`.  
   Greedy decoding + robust JSON parsing with regex fallback.

**Results**: Strong candidates score ~0.9–1.0; explicit HR intent dominates rankings.

### Part II – Fine-Tuning for Better Calibration

1. **Setup**  
   Install & import PEFT, TRL, Accelerate, bitsandbytes.

2. **Model Loading**  
   Load base causal LLM (likely Llama family) with quantization.

3. **Parameter-Efficient Fine-Tuning (PEFT)**  
   Apply **LoRA** adapters (low-rank updates).  
   Use **SFTTrainer** for supervised fine-tuning on task-specific examples.

4. **Training**  
   1 epoch (~8 min on T4 GPU).  
   Training loss converges (~0.91).

5. **Inference & Evaluation**  
   Generate fit scores post-fine-tuning.  
   Observe improved nuance: scores shift from overconfident (many 1.0) to calibrated (0.80–0.95 range).

**Results**: Fine-tuned model provides more discriminating, recruiter-friendly rankings.

## Key Learnings & Techniques Demonstrated

- Multi-method comparison: classical ML → embeddings → generative LLMs.
- Prompt engineering for structured JSON output from LLMs.
- Efficient fine-tuning (LoRA + 4/8-bit quantization) on consumer GPU.
- Handling noisy real-world text data (job titles, locations).
- Calibration improvement through domain-specific adaptation.

## Results Highlights

- **Baseline (Part I)**: High precision on explicit HR profiles (fit > 0.9).  
- **Fine-tuned (Part II)**: Better ranking granularity and reduced overconfidence.  
- Top candidates consistently include "Aspiring Human Resources", "HR Generalist", etc.

## Conclusion & Value for Recruiters / Hiring Managers

This project showcases my ability to design, implement, and iteratively improve an **end-to-end AI-powered talent sourcing tool** — a highly relevant skill in modern HR tech, recruiting platforms, and talent marketplaces.

Contributions include:
- Rapid prototyping of multiple NLP approaches to solve a real business problem.
- Transition from heuristic + embedding methods to state-of-the-art LLM prompting.
- Successful application of **PEFT/LoRA** for efficient model adaptation on limited hardware.
- Production-ready considerations: quantization, error handling, structured outputs.

The result is a practical, explainable system that can prioritize high-potential HR candidates from large applicant pools — reducing manual screening time while maintaining fairness and relevance.

Feel free to explore the notebooks, run them in Colab (GPU recommended), or reach out for questions/discussion. I'm excited to apply similar techniques to real-world talent challenges!

**Hongyu**  
Canberra, Australia  
