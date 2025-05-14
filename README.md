~AI agent的第三份作業 ~
# 🎓 English Learning AI Assistant  
Multi-Agent + Chroma RAG + Flask Web System  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4o-green)](https://openai.com/)  
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Database-orange)](https://docs.trychroma.com/)  

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Introduction
This project develops an intelligent English learning assistant integrating **AI Multi-Agent** and **Chroma Vector Database (RAG)**.  
It allows students to query knowledge from uploaded PDFs and a question bank, automatically generate exercises, and receive feedback.

The system simulates:  
- Agent1: English Assistant (explain concepts)  
- Agent2: Test Designer (generate 5 multiple choice exercises)  
- Agent3: Auto Grader (correct student answers and explain)

> 📝 Originally designed for junior high school English learning.

---

## 🏗️ System Architecture
```plaintext
User ➡️ Agent1 (Answer with RAG) ➡️ Agent2 (Generate exercises) ➡️ User
                                 ➕
                        (Session memory for follow-up correction)

# 🎓 English Learning AI Assistant  
Multi-Agent + Chroma RAG + Flask Web System  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4o-green)](https://openai.com/)  
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Database-orange)](https://docs.trychroma.com/)  

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Introduction
This project develops an intelligent English learning assistant integrating **AI Multi-Agent** and **Chroma Vector Database (RAG)**.  
It allows students to query knowledge from uploaded PDFs and a question bank, automatically generate exercises, and receive feedback.

The system simulates:  
- Agent1: English Assistant (explain concepts)  
- Agent2: Test Designer (generate 5 multiple choice exercises)  
- Agent3: Auto Grader (correct student answers and explain)

> 📝 Originally designed for junior high school English learning.

---

## 🏗️ System Architecture
```plaintext
User ➡️ Agent1 (Answer with RAG) ➡️ Agent2 (Generate exercises) ➡️ User
                                 ➕
                        (Session memory for follow-up correction)
