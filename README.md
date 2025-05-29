# 🤖 SmartResolve: AI-Powered Customer Support Assistant (RAG + IBM watsonx.ai)

SmartResolve is a proof-of-concept AI solution built for the IBM watsonx.ai RAG Hackathon, under the theme **"Advance the future of customer experience."** It combines **Retrieval-Augmented Generation (RAG)** using **ChromaDB** and **IBM watsonx.ai** to deliver a smarter, more accurate virtual support assistant that enhances how users interact with customer service systems.

---

## ✅ Deliverable 1: Video Demonstration

🎥 Watch the 3-minute demo here: [Demo](https://youtu.be/TS0Qy36nLio))

- Shows document upload
- Demonstrates live question-answering using retrieved KB
- Narrated walkthrough of Chroma retrieval + watsonx.ai generation

---

## ✅ Deliverable 2: Written Problem and Solution Statement

### Problem

Modern customer service often struggles with slow response times, generic chatbot replies, and long wait queues. This leads to poor customer satisfaction, frustration, and churn. While LLMs can answer general questions, they often hallucinate or lack the specific business context needed for personalized help.

### Solution

**SmartResolve** bridges that gap by combining your company’s internal knowledge base (KB) with the power of generative AI. It uses **Retrieval-Augmented Generation (RAG)** to fetch relevant content from your custom FAQ and then calls **IBM watsonx.ai** to generate accurate, human-like answers.

#### Target Users
- End-users and customers seeking self-service support
- Companies looking to enhance virtual agent capabilities

#### How It Works
1. Users upload a `.txt` FAQ or KB file.
2. The app stores embeddings in **Chroma**, a local vector database.
3. When a user asks a question, Chroma retrieves the top relevant chunks.
4. The retrieved context is sent to **IBM watsonx.ai** (Granite-13B chat model) to generate an accurate, fluent response.

#### Why It’s Creative
- No training required — users just upload text!
- Runs fully on local infrastructure + cloud inference
- Leverages cutting-edge RAG design with open-source tooling
- Minimal setup, fully streamlit-based, easy to deploy and demo

---

## ✅ Deliverable 3: RAG and watsonx.ai Usage

### Retrieval-Augmented Generation (RAG)

- **Retriever**: `ChromaDB` stores vectorized embeddings of support content using `SentenceTransformers`.
- **Augmenter**: When a question is asked, Chroma retrieves the most relevant KB lines.
- **Generator**: These lines are included as context in a prompt to **IBM watsonx.ai**.

### IBM watsonx.ai Integration

- The app uses `ibm-watsonx-ai` SDK to connect to **Granite-13B-chat** model.
- Custom prompts combine real-time user questions and retrieved knowledge base chunks.
- `generate_text()` is called with max token and temperature configs.

---

## ✅ Deliverable 4: Code Repository

🗂 GitHub Repo: [https://github.com/rahul-bhave/SmartResolve](https://github.com/rahul-bhave/SmartResolve)

## ✅ Deliverable 5: Local Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-smartresolve.git
cd rag-smartresolve
```
### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Setup environment variables for Watsonx AI

```bash
export WATSONX_API_KEY="your_ibm_api_key"
export WATSONX_PROJECT_ID="your_project_id"
```

### 5. Run Streamlit app
```bash
streamlit run app.py
```






