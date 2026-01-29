# DEVELOPMENT OF INFORMATION TAGGING  MODEL FOR HADITH STRUCTURE DATA INTO KNOWLEDGE GRAPH

> **Final Year Project (FYP)**  
> Bachelor of Applied Science in Data Analytics (Hons)  
> Universiti Malaysia Pahang Al-Sultan Abdullah

---

## 📌 Project Overview

Hadith texts contain rich, multi-layered information such as **narrators, themes, sources, and contextual relationships**. However, traditional keyword-based search systems struggle to capture these complex relationships.

This project proposes an **automated information tagging model** that transforms **structured Hadith data into a Knowledge Graph**, enabling **semantic understanding, relational querying, and advanced retrieval** of Hadith knowledge.

By combining **Natural Language Processing (NLP)** techniques with **graph databases**, this system provides a modern and scalable approach to digital Hadith studies.

---

## 🎯 Objectives

- Develop an **information tagging model** for Hadith texts  
- Automatically identify:
  - Narrators
  - Themes
  - Sources
  - Relationships between Hadith entities
- Construct a **Knowledge Graph** representation of Hadith data
- Evaluate the tagging model using standard NLP metrics

---

## 🧠 Key Technologies & Methods

- **Natural Language Processing (NLP)**
  - Named Entity Recognition (NER)
  - Regular Expressions
- **Data Tagging Model**
  - Narrator extraction
  - Theme classification
- **Knowledge Graph**
  - Neo4j Graph Database
  - Entity–relationship modeling
- **Evaluation Metrics**
  - Precision
  - Recall
  - F1-Score
- **Deployment**
  - Interactive Hadith Knowledge Graph Web Application

---

## 🗂 Dataset

- **Source**: https://sunnah.com  
- **Total Hadiths**: 45,346 English-translated Hadiths  
- **Collections Include**:
  - Sahih Muslim
  - Sahih Bukhari
  - Sunan Abi Dawud
  - and others

Data was collected using **web scraping**, then cleaned, normalized, and merged before tagging.

---

## 📊 Model Performance

The tagging model was evaluated against **human-extracted ground truth**:

| Metric     | Score |
|-----------|-------|
| Precision | 0.646 |
| Recall    | 0.663 |
| F1-Score  | 0.649 |

These results indicate that the model performs **effectively and consistently** for automated Hadith knowledge extraction.

---

## 🌐 Web Application Features

- Interactive Hadith Knowledge Graph visualization
- Relationship-based querying using Cypher
- Hadith retrieval by:
  - Narrator
  - Theme
  - Source
- Tabular relationship inspection

---

## 🚀 Significance of the Project

- 📚 Modernizes Hadith studies using AI and data analytics  
- 🧠 Enables **semantic search** instead of keyword-only search  
- 🔗 Reveals hidden relationships between Hadith narrations  
- 🏗 Provides a scalable framework for:
  - Quranic studies
  - Tafsir analysis
  - Islamic knowledge representation

---

## 🛠 Tools & Libraries

- Python
- Neo4j
- NLP (NER, Regex)
- Streamlit
- Pandas, NumPy

---

## ✍️ Work in Progress 

- Standardizing the file
- Containerize in docker
- Final adjustment to the filename and file in code
- Updating README
