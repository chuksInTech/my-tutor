MyTutor — Your AI-Powered Learning Assistant


MyTutor is a powerful, stream-based, interactive command-line AI assistant that uses both **OpenAI** and **Ollama** models to help users learn programming, academic subjects, and general knowledge effectively.

## 🚀 Features


-  **Streamed responses** from both GPT (OpenAI) and Ollama (LLaMA).
-  
-  Categorizes questions into **coding**, **academic**, and **general** types.
  
-  Provides **tailored answers** with system prompts based on question type.
  
-  Suggests **visual learning resources** when relevant.
   
-  Uses `.env` file for safe configuration of keys and URLs.
   

##  Requirements

- Python 3.8+
- [Ollama](https://ollama.com/)
- OpenAI API key
- `pip` for installing dependencies

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/my-tutor.git

cd my-tutor

. Install Dependencies

pip install -r requirements.txt

Add the following content to .env

OPENAI_API_KEY=your_openai_api_key

OLLAMA_BASE_URL=http://localhost:11434

Make sure you have Ollama running locally with the model installed, e.g., ollama run llama3.

. Run MyTutor

python mytutor.py

Got any question? feel free to contact me: chyootch@gmail.com
