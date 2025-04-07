# 🤖 Smart AI Resume Parser & Job Matching System

An intelligent AI-powered Resume Parser and Job Matching System that leverages NLP, Deep Learning, and Graph Neural Networks to analyze resumes, extract structured information, and match candidates with the most relevant job opportunities.

## 🚀 Features

- 📄 Upload and parse resumes (PDF supported)
- 🔍 Extract candidate information (Skills, Education, Experience, Projects, etc.)
- 🧠 Match resumes to job descriptions using:
  - NLP (SpaCy, Sentence-BERT)
  - Graph Neural Networks (via NetworkX)
- 📊 Provide candidate-job fit score
- 🌐 Web interface with seamless user experience
- 🗂 Resume builder & suggestions

## 🛠 Tech Stack

- **Frontend**: Streamlit / HTML UI components
- **Backend**: Python, Flask / Streamlit
- **NLP**: SpaCy, PyMuPDF, Sentence-BERT
- **Graph Analysis**: NetworkX
- **Database**: SQLite
- **AI Models**: Custom-trained DL models
- **Utilities**: OpenAI Embeddings (optional), Excel Export


 📦 Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/DHANUSHRAJA22/-Resume-Parser-Job-Matching.git
   cd -Resume-Parser-Job-Matching
Create and activate virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate       # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
💡 Future Enhancements
🔒 Add authentication

☁️ Cloud deployment (e.g., Streamlit Cloud, Render, AWS)

🧾 OCR-based parsing for scanned resumes

📈 Admin dashboard with analytics
