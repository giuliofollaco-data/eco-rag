from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retrieved_context

model = OllamaLLM(model="llama3.2")


# ── 1. Prompt d'expansion de requête ───────────────────────────────────────────

EXPAND_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un expert en recherche documentaire sur les rapports scientifiques du GIEC.
Ta tâche est de formuler exactement 3 requêtes de recherche complémentaires, \
une par ligne, sans numérotation, sans explication, sans texte autour.

Les 3 requêtes doivent couvrir :
1. La donnée précise demandée (chiffre, mesure, date)
2. La conclusion générale du rapport sur ce sujet (résumé, bilan, certitude)
3. Le niveau de confiance ou de certitude scientifique associé

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question : {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ── 2. Prompt de génération de réponse ─────────────────────────────────────────

ANSWER_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant expert en science du climat, spécialisé dans le rapport de \
synthèse 2023 du GIEC (AR6 SYR).

Réponds à la question en utilisant UNIQUEMENT les extraits fournis dans la section "CONTEXTE".

Règles strictes — respecte-les dans cet ordre de priorité :

1. HIÉRARCHIE DES SOURCES
   • Les extraits marqués [SPM] (Résumé pour Décideurs) représentent la \
conclusion officielle et synthétique du rapport. Ils ont la priorité absolue \
sur les extraits du [CORPS] pour toute affirmation générale ou conclusion.
   • Les extraits du [CORPS] apportent le détail technique. Utilise-les pour \
préciser ou illustrer, jamais pour contredire le SPM.

2. ARBITRAGE ENTRE CHIFFRES SIMILAIRES
   • Si plusieurs valeurs proches apparaissent dans le contexte (ex : 1,07 / \
1,1 / 1,15 °C), retiens UNIQUEMENT celle qui est explicitement présentée \
comme la valeur de référence principale dans un extrait [SPM] ou dans la \
phrase de conclusion d'un extrait [CORPS].
   • Mentionne l'existence des autres valeurs en précisant leur signification \
(ex : « valeur pour une période légèrement différente »).

3. NIVEAU DE CONFIANCE
   • Conserve systématiquement les qualificatifs du GIEC : \
"confiance élevée", "confiance très élevée", "probabilité virtuelle", etc.
   • Ne reformule pas ces termes techniques — ils ont une définition précise.

4. CITATION DES SOURCES
   • Pour chaque affirmation chiffrée ou conclusion importante, indique \
entre crochets la page source, ex : [p. 12, SPM].

5. ABSENCE DE RÉPONSE
   • Si la réponse n'est pas dans le contexte, réponds exactement :
     "Les extraits fournis ne permettent pas de répondre à cette question."
   • Ne fais jamais appel à tes connaissances externes.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXTE :
{context}

---

QUESTION :
{question}

Réponse en français :<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

expand_prompt = ChatPromptTemplate.from_template(EXPAND_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

expand_chain = expand_prompt | model
answer_chain = answer_prompt | model


# ── Fonctions utilitaires ───────────────────────────────────────────────────────


def expand_queries(question: str) -> list[str]:
    """Génère des sous-requêtes complémentaires via le LLM."""
    raw = expand_chain.invoke({"question": question})
    sub_queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    return [question] + sub_queries[:3]


def gather_context(question: str, k_per_query: int = 5) -> str:
    """
    Multi-query retrieval :
    - Lance un retrieval pour chaque sous-requête
    - Déduplique les chunks entre les requêtes
    - Retourne au maximum 12 chunks pour ne pas dépasser la fenêtre de contexte
    """
    queries = expand_queries(question)
    print(f"[INFO] {len(queries)} requête(s) utilisées :")
    for i, q in enumerate(queries):
        print(f"  [{i+1}] {q}")

    seen, all_chunks = set(), []
    for q in queries:
        context_block = get_retrieved_context(q, k=k_per_query)
        for chunk in context_block.split("\n\n---\n\n"):
            key = chunk[:120]
            if key not in seen:
                seen.add(key)
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

        print("→ Expansion de la requête...")
        print("→ Recherche dans le rapport...")
        context = gather_context(question)

        print("→ Génération de la réponse...\n")
        response = answer_chain.invoke({"context": context, "question": question})
        print(response)
        print("\n" + "─" * 60)
