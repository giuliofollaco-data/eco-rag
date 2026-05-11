from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Tu es un assistant expert en science du climat, spécialisé dans l'analyse du rapport de synthèse 2023 du GIEC (AR6). 

    Ton rôle est de répondre à la question de l'utilisateur en utilisant UNIQUEMENT les extraits du rapport fournis dans la section "CONTEXTE". 

    Consignes strictes :
    1. Si la réponse ne se trouve pas dans le contexte fourni, dis-le clairement : "Je suis désolé, mais les extraits du rapport fournis ne me permettent pas de répondre à cette question."
    2. Ne fais pas appel à tes connaissances externes.
    3. Conserve la précision scientifique (niveaux de confiance, chiffres exacts).
    4. Cite les sections ou les faits mentionnés dans le contexte.
    5. Réponds de manière concise et structurée.

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    CONTEXTE :
    {context}

    ---

    QUESTION : 
    {question}

    Réponse en français :<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input(
        "Posez votre question sur le rapport du GIEC 2023 (ou tapez 'q' pour quitter) : "
    )
    print("\n\n")
    if question.lower() == "q":
        break

    context = None

    response = chain.invoke({"context": context, "question": question})
    print(response)
    print("\n" + "-" * 50 + "\n")
