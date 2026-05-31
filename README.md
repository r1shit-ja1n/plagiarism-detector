# 🕵️‍♂️ Plagiarism Detector

An advanced C++ plagiarism detection engine with a Streamlit UI.

## 📖 Overview
This project is a hybrid-architecture plagiarism detection platform. It combines the raw computational speed and memory management of a native **C++17 backend** with the modern, interactive frontend capabilities of **Python and Streamlit**. It uses advanced Natural Language Processing (NLP) techniques and deterministic string-matching algorithms to analyze, compare, and score the similarity between documents.

## 💡 Why This Exists & Who It Benefits
In an era of AI-generated content and massive digital publishing, verifying the originality of text is critical. This tool is designed to benefit:
* **Educators & Teaching Assistants:** Quickly batch-process student essays or code to check for unauthorized copying.
* **Content Creators & Publishers:** Verify that articles and blogs are entirely original before publishing to avoid SEO penalties.
* **Developers & Data Scientists:** Utilize a lightweight, lightning-fast text deduplication engine without relying on expensive, rate-limited cloud APIs.

## 🏆 The Competitive Advantage: Why This is Different
Most open-source plagiarism detectors are written purely in Python. While Python is great for prototyping, it struggles with performance when comparing massive datasets or thousands of documents. 

**This engine solves that by using a "Dual-Architecture" approach:**
* **Native C++ Speed:** The heavy lifting (tokenization, n-gram generation, and dynamic programming) is written entirely in C++17, compiling directly to machine code for maximum execution speed and zero garbage-collection overhead.
* **No Black-Box APIs:** Unlike cloud-based checkers, this runs 100% locally. It uses mathematically proven, deterministic algorithms, guaranteeing absolute data privacy.
* **Multi-Algorithm Verification:** Instead of relying on a single metric, the engine cross-references text using **four** different computer science algorithms to prevent false positives.
* **Seamless UI Bridge:** A custom Python subprocess bridge connects the C++ binary to a sleek, dark-themed Streamlit web interface, making it accessible to non-technical users.

## 🧠 Core Algorithms Implemented
The C++ engine mathematically analyzes text overlap using the following techniques:
* **Jaccard Similarity (N-Grams):** Chunks text into sliding windows of $N$ words and calculates the exact intersection over union, effectively catching paraphrased plagiarism.
* **Knuth-Morris-Pratt (KMP):** Utilizes Longest Proper Prefix (LPS) tables to skip redundant string comparisons for $O(N+M)$ exact block-match detection.
* **Rabin-Karp:** Deploys rolling polynomial hashes to instantly detect exact copied phrases in $O(1)$ window sliding time.
* **Longest Common Subsequence (LCS):** Uses Dynamic Programming (DP) to find the longest shared sequence of words even if they are interrupted by new, inserted text.

## 💻 Tech Stack
* **Backend Engine:** C++17 (Standard Template Library)
* **Frontend UI:** Python 3, Streamlit
* **Bridge:** Python `subprocess` routing

## ⚙️ How to Run Locally

### 1. Prerequisites
Ensure you have a C++ compiler (like `g++`) and Python installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/r1shit-ja1n/plagiarism-detector.git](https://github.com/r1shit-ja1n/plagiarism-detector.git)
cd plagiarism-detector
