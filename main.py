from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retrieved_context

model = OllamaLLM(model="llama3.2", temperature=0.0)

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

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
answer_chain = answer_prompt | model


# ── Fonctions utilitaires ───────────────────────────────────────────────────────


def gather_context(question: str, k: int = 5) -> str:
    """
    Retrieval :
    - Lance un retrieval pour la requête
    - Retourne au maximum 12 chunks pour ne pas dépasser la fenêtre de contexte
    """

    all_chunks = []
    context_block = get_retrieved_context(question, k)
    for chunk in context_block.split("\n\n---\n\n"):
        all_chunks.append(chunk)

    print(
        f"[INFO] {len(all_chunks)} fragments uniques récupérés (max 12 envoyés au LLM)."
    )
    return "\n\n---\n\n".join(all_chunks[:12])


# ── Boucle principale ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Assistant GIEC AR6 — v2")
    print("=" * 60)

    while True:
        question = input(
            "\nVotre question sur le rapport du GIEC 2023 (ou 'q' pour quitter) : "
        ).strip()
        print()

        if question.lower() == "q":
            print("Au revoir.")
            break

        print("→ Recherche dans le rapport...")
        context = gather_context(question)

        print("→ Génération de la réponse...\n")
        response = answer_chain.invoke({"context": context, "question": question})
        print(response)
        print("\n" + "─" * 60)
