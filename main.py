import os
from datetime import datetime

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retrieved_context

model = OllamaLLM(model="llama3.2", temperature=0.0)

# ── Prompt de traduction de question ─────────────────────────────────────────

QUERY_TRANSLATION_TEMPLATE = """
Tu es un traducteur expert en sciences du climat, spécialisé dans la terminologie \
officielle du GIEC (IPCC).

Ta tâche est de traduire la question de l'utilisateur du français vers l'anglais. \
Cette traduction sera directement injectée dans un moteur de recherche hybride (dense + lexical).

Règles strictes à respecter :
1. Terminologie GIEC : Utilise rigoureusement le vocabulaire officiel du rapport de synthèse \
(ex: utilise "contexts that are highly vulnerable", "low-emissions pathways", "climate-resilient \
development", etc.).
2. Pas de fioritures : Génère UNIQUEMENT la traduction de la question. Ne commence \
jamais ta réponse par "Voici la traduction :", "Sure, here is the translation", ou \
toute autre phrase d'introduction. Pas de commentaires.

Question en français : {question}

Traduction en anglais (Strictement la question seule) :
"""

# ── Prompt de génération de réponse ─────────────────────────────────────────

ANSWER_TEMPLATE = """
Tu es un assistant expert en science du climat, spécialisé dans le rapport de \
synthèse 2023 du GIEC (AR6 SYR).

Réponds à la question en utilisant UNIQUEMENT les extraits fournis dans la section "CONTEXTE".

Règles strictes — respecte-les dans cet ordre de priorité :

1. HIÉRARCHIE DES SOURCES
- Les extraits marqués [SPM] (Résumé pour Décideurs) représentent la \
conclusion officielle et synthétique du rapport. Ils ont la priorité absolue \
sur les extraits du [CORPS] pour toute affirmation générale ou conclusion.
- Les extraits du [CORPS] apportent le détail technique. Utilise-les pour \
préciser ou illustrer, jamais pour contredire le SPM.

2. ARBITRAGE ENTRE CHIFFRES SIMILAIRES
- Si plusieurs valeurs proches apparaissent dans le contexte (ex : 1,07 / \
1,1 / 1,15 °C), retiens UNIQUEMENT celle qui est explicitement présentée \
comme la valeur de référence principale dans un extrait [SPM] ou dans la \
phrase de conclusion d'un extrait [CORPS].
- Mentionne l'existence des autres valeurs en précisant leur signification \
(ex : « valeur pour une période légèrement différente »).

3. TERMINOLOGIE DU GIEC
- Niveaux de confiance : confiance très faible / faible / moyenne / élevée / très élevée
- Probabilités : pratiquement certain / très probable / probable / aussi probable qu'improbable
- Si les preuves se sont "renforcées depuis l'AR5", le mentionner explicitement.

4. CITATION DES SOURCES
- Pour chaque affirmation chiffrée ou conclusion importante, indique \
entre crochets la page source, ex : [p. 12, SPM].

5. ABSENCE DE RÉPONSE
- Si la réponse n'est pas dans le contexte, réponds exactement :
"Les extraits fournis ne permettent pas de répondre à cette question."
- Ne fais jamais appel à tes connaissances externes.

CONTEXTE :
{context}

---

QUESTION :
{question}

Réponse en français :
"""

query_translation_prompt = ChatPromptTemplate.from_template(QUERY_TRANSLATION_TEMPLATE)
query_translation_chain = query_translation_prompt | model

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
answer_chain = answer_prompt | model


# ── Boucle principale ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("-" * 50)
    print("  Assistant GIEC AR6")
    print("-" * 50)

    while True:
        question = input(
            "\nVotre question sur le rapport du GIEC AR6 ('q' pour quitter) : "
        ).strip()
        print()

        if question.lower() == "q":
            break

        # Traduction de la question pour le retrieval (.strip() pour éviter les espaces superflus)
        question_translated = query_translation_chain.invoke(
            {"question": question}
        ).strip()
        print(f"[INFO] Question traduite pour le retrieval : {question_translated}")

        # Retrieval du contexte pertinent dans le rapport (stratégie hybride + injection SPM)
        print("Recherche dans le rapport...")
        context = get_retrieved_context(question_translated)

        # Génération de la réponse à partir du contexte et de la question originale
        print("Génération de la réponse...\n")
        response = answer_chain.invoke({"context": context, "question": question})
        print(response)
        print("\n" + "─" * 50)
