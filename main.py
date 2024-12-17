import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain


# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


# Obtém a chave de API da variável de ambiente
google_api_key = os.getenv("GOOGLE_API_KEY")


# Verifica se a chave de API foi carregada corretamente
if not google_api_key:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não está definida. Certifique-se de ter um arquivo .env com a chave correta.")

# Inicializa o modelo Google Generative AI usando a chave carregada.
llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7, google_api_key=google_api_key)

# Cria um prompt template.
template = """Pergunta: {pergunta}

Resposta:"""
prompt = PromptTemplate(template=template, input_variables=["pergunta"])

# Cria uma LLMChain, que combina o modelo de linguagem com o prompt.
llm_chain = LLMChain(prompt=prompt, llm=llm)

perguntas = [
    "Qual a capital do Brasil?",
    "Quem escreveu Dom Casmurro?",
]

for pergunta in perguntas:
    resposta = llm_chain.run(pergunta)
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}\n")
