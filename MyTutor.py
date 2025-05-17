#!/usr/bin/env python
# coding: utf-8

# # End of week 1 exercise
# 
# To demonstrate your familiarity with OpenAI API, and also Ollama, build a tool that takes a technical question,  
# and responds with an explanation. This is a tool that you will be able to use yourself during the course!

# In[1]:


# imports
import os
import json
import logging
from typing import Dict, List, Optional, Union
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display
import openai


# In[2]:


# load environment variables

load_dotenv()

#constants

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

GPT_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3.2"

#configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MyTutor")


# In[3]:


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str, callback, system_prompt: str = None) -> str:
        """Generate a stream response from ollama. A callback function is required to handle streamed chunks
            Args:
                prompt (str): User's prompt
                system_prompt (str, optional): System-level prompt
                callback (function): Required function to process each streamed chunk

        Returns:
            None
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode("utf-8"))
                    chunk = line_data.get("response", "")
                    callback(chunk)

                    if line_data.get("done", False):
                        break

        except Exception as e:
            logger.error(f"Error generating response from ollama: {e} ")
            return f"Error: {e} "


class OpenAIClient:
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = GPT_MODEL) :
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, prompt: str, callback, system_prompt: str = None) -> str :
        """Generate a streaming response from OpenAI. Callback function is required to process streamed chunks
            Args:
                prompt (str): User's prompt
                system_prompt (str, optional): System-level prompt
                callback (function): Required function to process each streamed chunk

        Returns:
            None
        """

        try:
            messages = []
            if system_prompt:
                messages.append({"role" : "system", "content" : system_prompt})
            messages.append({"role" : "user", "content" : prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                stream=True
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        callback(content)

            return ""

        except Exception as e:
            logger.error(f"Error generating response from  OpenAI: {e}")
            return f"Error: {e} " 


# In[4]:


class MyTutor:
    def __init__(
        self,
        use_ollama: bool = True,
        use_gpt: bool = True,
        ollama_model: str = OLLAMA_MODEL,
        gpt_model: str = GPT_MODEL
    ):
        self.use_ollama = use_ollama
        self.use_gpt = use_gpt

        # Initialize clients based on configuration

        if self.use_ollama:
            self.ollama_client = OllamaClient(model=ollama_model)

        if self.use_gpt:
            self.openai_client = OpenAIClient(model=gpt_model)

        # Define the system prompt for "My Tutor"

        self.system_prompt = """You are 'My Tutor', a versatile and knowledgeable assistant.

            YOUR ROLE:
            - Help with programming and technical questions across various languages and frameworks
            - Answer academic questions on subjects like math, science, history, literature, etc.
            - Provide explanations on general knowledge topics
            - Assist with writing, research, and conceptual understanding
            - Help with learning strategies and educational resources

            WHEN RESPONDING:
            1. First understand the question's domain and specific requirements
            2. For coding and technical questions:
               - Break down solutions step by step
               - Provide complete, working code with appropriate comments
               - Explain key concepts and best practices
            3. For academic questions:
               - Explain concepts clearly using appropriate terminology
               - Provide relevant examples and applications
               - Structure information in a learning-friendly format
            4. For general knowledge:
               - Present balanced, factual information
               - Provide context and background when helpful
            5. For all responses:
               - Be clear, concise, and educational
               - Cite general sources of information when relevant
               - Admit if you don't know something rather than guessing

            You're a helpful guide focused on providing accurate, educational responses to help the user learn and understand.
            Respond in Markdown
            """

    def detect_question_type(self, question: str) -> str:
        """
        Detect the general category of the question to optimize response.

        Returns:
            String indicating question type: "coding", "academic", "general"
        """
        # Simple keyword-based detection - could be improved with ML/better heuristics
        coding_keywords = [
            "code", "programming", "function", "class", "bug", "error",
            "python", "javascript", "java", "html", "css", "sql",
            "algorithm", "compile", "debug", "syntax", "library"
        ]

        academic_keywords = [
            "math", "science", "history", "literature", "biology", "chemistry",
            "physics", "equation", "theorem", "theory", "calculate", "solve",
            "essay", "analysis", "research", "study", "homework"
        ]

        # Check for coding-related content
        if any(keyword in question.lower() for keyword in coding_keywords):
            return "coding"
        # Check for academic content
        elif any(keyword in question.lower() for keyword in academic_keywords):
            return "academic"
        # Default to general
        else:
            return "general"


    def ask(self, question: str, use_model: str = "ollama", output_callback=None) -> Dict[str, str] :
        """
        Ask a question to the tutor and get responses.

        Args:
            question: The user's question
            use_model: Which model to use - "ollama", "gpt", or both"

        Returns:
            Dictionary containing streamed responses from the requested models
        """

        if output_callback is None:
            raise ValueError("output_callback is required for streaming responses")


        # Detect question type
        question_type = self.detect_question_type(question)

        # Customize system prompt based on question type
        customized_prompt = self.system_prompt
        if question_type == "coding":
            customized_prompt += "\nThis appears to be a coding or technical question. Focus on providing clear, executable code and technical explanations."
        elif question_type == "academics":
            customized_prompt += "\nThis appears to be an academic question. Focus on providing educational, structured explanations with relevant examples."


        #stream from ollama
        if (use_model in ["ollama", "both"]) and self.use_ollama:
            def ollama_callback(chunk):
                output_callback("ollama", chunk, False)

            self.ollama_client.generate_response(
                prompt=question,
                system_prompt=customized_prompt,
                callback=ollama_callback
            )
            output_callback("ollama", "", True)

        #stream for openai
        if (use_model in ["gpt", "both"]) and self.use_gpt:
            def gpt_callback(chunk):
                output_callback("gpt", chunk, False)

            self.openai_client.generate_response(
                prompt=question,
                callback=gpt_callback,
                system_prompt=customized_prompt
            )
            output_callback("gpt", "", True)


        if question_type in ["academics", "coding"]:
            visual_suggestions = self.suggest_visual_resources(question)

            if visual_suggestions:
                if use_model in ["ollama", "both"] and self.use_ollama:
                    output_callback("ollama", visual_suggestions, True)
                if use_model in ["gpt", "both"] and self.use_gpt:
                    output_callback("gpt", visual_suggestions, True)
                for model in responses:
                    responses[model] += visual_suggestions


    def suggest_visual_resources(self, topic: str) -> str:
        """Suggest visual resources for learning about a topic."""
        visual_resources = {
            "math": ["Desmos Graphing Calculator", "GeoGebra", "Khan Academy interactive visualizations"],
            "physics": ["PhET Interactive Simulations", "The Physics Classroom Animations"],
            "chemistry": ["Molecular 3D Viewers", "Periodic Table Interactive"],
            "biology": ["Cell Interactive Diagrams", "BioDigital Human Explorer"],
            "computer science": ["Algorithm Visualizations", "Compiler Explorer", "Python Tutor"],
            "programming": ["Visual Studio Code", "PyCharm", "GitHub Copilot"],
            "data science": ["Jupyter Notebook", "Tableau Public", "Google Colab"],
            "machine learning": ["TensorBoard", "Weights & Biases", "Neural Network Playground"],
            "history": ["Interactive Timeline Tools", "Map Visualization Tools"],
            "language learning": ["Duolingo", "Babbel", "Memrise"],
            "art": ["Canva", "Procreate", "Blender"],
            "design": ["Figma", "Adobe XD", "Sketch"],
            "general": ["YouTube Educational Channels", "Coursera", "edX"]
        }

        matched_resources = []
        topic_words = topic.lower().split()
        for subject, resources in visual_resources.items():
            if any(word in subject for word in topic_words):
                matched_resources.extend(resources)

        if matched_resources:
            resource_list = "\n- ".join(matched_resources)
            return f"\nVisual Learning Resources:\n- {resource_list}\n"

        return ""

    def format_response(self, response: str, question_type: str) -> str:
        """Format the response based on the question type."""
        if question_type == "coding":
            return response
        elif question_type == "academics":
            if len(response) > 500:
                sections = response.split(". ")

                if len(sections) > 5:
                    formatted = "# Summary\n\n"
                    formatted += sections[0] + ".\n\n"
                    formatted += "# Detailed Explanation\n\n"
                    formatted += ". ".join(sections[1:])
                    return formatted

                return response
            else:
                return response



# In[5]:


def main():
    print("Welcome to My Tutor - Your All-Purpose Learning Assistant!")
    print("Type 'exit' to quit, 'switch' to change models, or ask any question.\n")

    # Always use streamed responses
    tutor = MyTutor()
    current_model = "ollama"  # Default to using both models

    while True:
        user_input = input("\n🧑💻 You: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye! Hope I helped!")
            break

        elif user_input.lower() == "switch":
            print("\nSelect model: 'ollama', 'gpt', 'both'")
            model_choice = input("Model choice: ").lower()
            if model_choice in ["ollama", "gpt", "both"]:
                current_model = model_choice
                print(f"Switched to using {current_model} model(s)")
            else:
                print("Invalid choice. Using previous setting.")
            continue

        # Always stream output
        response_buffers = {"ollama": "", "gpt": ""}
        active_models = set()

        def handle_streaming_output(model, chunk, done):
            nonlocal response_buffers, active_models

            if model not in active_models and not done:
                active_models.add(model)
                print(f"\n {model.capitalize()} says:")

            if not done:
                response_buffers[model] += chunk
                print(chunk, end="", flush=True)

            if done and model in active_models:
                print()  # Print newline when done

        # Use streaming method only
        tutor.ask(user_input, use_model=current_model, output_callback=handle_streaming_output)


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


# Get gpt-4o-mini to answer, with streaming


# In[ ]:


# Get Llama 3.2 to answer

