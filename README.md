# IPCC AR6 — Moteur de question-réponse RAG local

> Interroger 10 ans de science climatique en langage naturel, sans envoyer une seule donnée dans le cloud.

---

## Aperçu

Ce projet implémente un système **RAG** (*Retrieval-Augmented Generation*) entièrement souverain pour interroger le **rapport de synthèse 2023 du GIEC (AR6 SYR)** — 220 pages de conclusions scientifiques, tableaux et niveaux de confiance. L'utilisateur pose une question en français ; le moteur retrouve les passages pertinents dans le document et génère une réponse sourcée, avec citation de page, sans jamais faire appel à un service externe.

---

## Problématique

Les LLM généralistes hallucinent sur les données chiffrées et perdent le contexte des nuances de probabilité propres aux rapports du GIEC. Comment garantir la **fidélité factuelle** d'un chatbot spécialisé sur un document volumineux et dense, tout en opérant de manière **100 % locale** ?

---

## Architecture

```
Question utilisateur
        │
        ▼
┌───────────────────────┐
│ Multi-Query Expansion │  → 3 sous-requêtes (donnée précise / conclusion / confiance)
└──────────┬────────────┘
           │
     ┌─────▼──────────────────────────────────┐
     │         Retrieval hybride              │
     │  ┌─────────────┐  ┌──────────────────┐ │
     │  │  Dense (60%)│  │   BM25   (40%)   │ │
     │  │  ChromaDB   │  │  (termes exacts) │ │
     │  └──────┬──────┘  └────────┬─────────┘ │
     │         └────────┬─────────┘           │
     │      Reciprocal Rank Fusion            │
     └─────────────────┬──────────────────────┘
                       │
           ┌───────────▼────────────┐
           │ Injection SPM garantie │  → ≥ 2 chunks du Résumé pour Décideurs
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │     LLM (llama3.2)     │  → Réponse sourcée, en français
           └────────────────────────┘
```

---

## Fonctionnalités clés

**Retrieval hybride dense + BM25**
La recherche vectorielle sémantique est combinée à BM25 via un algorithme de *Reciprocal Rank Fusion* (RRF) implémenté nativement. BM25 améliore la précision sur les termes exacts — chiffres, unités, années — là où la recherche dense seule produisait des confusions numériques (ex. : 1,07 vs 1,1 °C).

**Multi-query expansion**
Chaque question est automatiquement décomposée en 3 sous-requêtes complémentaires par le LLM, ciblant respectivement la donnée précise, la conclusion générale du rapport et le niveau de confiance associé. Les résultats sont dédupliqués et fusionnés.

**Hiérarchisation des sources (SPM > corps)**
Les chunks sont taggés à l'ingestion selon leur section d'origine : `SPM` (Résumé pour Décideurs, pages 1–54) ou `corps`. Au moins 2 chunks SPM sont injectés de force dans chaque contexte, garantissant que les conclusions officielles du rapport ne sont jamais occultées par un passage technique local.

**Pipeline entièrement local**
Aucune donnée ne quitte la machine. Le modèle d'embedding (`mxbai-embed-large`) et le LLM (`llama3.2`) tournent via Ollama ; la base vectorielle est persistée localement avec ChromaDB.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Orchestration | LangChain |
| LLM | Ollama — `llama3.2` |
| Embeddings | Ollama — `mxbai-embed-large` |
| Base vectorielle | ChromaDB |
| Retrieval lexical | BM25 (`rank-bm25`) |
| Langage | Python 3.10+ |
| Matériel de développement | RTX 3070 · 16 Go RAM |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/ipcc-ar6-rag.git
cd ipcc-ar6-rag

# 2. Installer les dépendances Python
pip install langchain langchain-community langchain-chroma langchain-ollama \
            langchain-text-splitters chromadb rank-bm25

# 3. Télécharger les modèles Ollama
ollama pull llama3.2
ollama pull mxbai-embed-large

# 4. Placer le PDF du rapport dans le répertoire racine
# Téléchargeable sur : https://www.ipcc.ch/report/ar6/syr/
cp /chemin/vers/IPCC_AR6_SYR_FullVolume.pdf .
```

---

## Utilisation

```bash
python main.py
```

La première exécution indexe le document et crée la base vectorielle (~10–20 min selon le matériel). Les runs suivants chargent la base depuis le cache.

```
Assistant GIEC AR6 — v2
════════════════════════════════════════════════════════════

Votre question sur le rapport du GIEC 2023 (ou 'q' pour quitter) :
→ Quelle est l'augmentation de la température mondiale observée entre 1850-1900 et 2011-2020 ?

→ Expansion de la requête...
→ Recherche dans le rapport...
→ Génération de la réponse...

La température moyenne mondiale à la surface du globe a augmenté de 1,1 °C
(intervalle probable : 1,0–1,2 °C) sur la période 2011-2020 par rapport
à 1850-1900 (confiance élevée) [p. 4, SPM].
```

> **Note Windows (PowerShell)** : pour réinitialiser la base après un changement de configuration, utiliser `Remove-Item -Recurse -Force chroma_db, chunks_cache.pkl`

---

## Structure du projet

```
ipcc-ar6-rag/
├── main.py                  # Boucle conversationnelle & prompts
├── vector.py                # Ingestion, indexation & retrieval hybride
├── IPCC_AR6_SYR_FullVolume.pdf
├── chroma_db/               # Base vectorielle persistée (généré)
└── chunks_cache.pkl         # Cache des chunks (généré)
```

---

## Choix techniques notables

**Pourquoi RRF natif plutôt qu'`EnsembleRetriever` ?**
`EnsembleRetriever` était initialement issu de `langchain.retrievers`, un module en cours de dépréciation. La fusion RRF est ré-implémentée nativement en ~30 lignes, ce qui supprime la dépendance à `langchain-classic` et rend l'algorithme de pondération directement lisible et modifiable.

**Pourquoi `num_ctx=4096` explicite sur Ollama ?**
Sans ce paramètre, Ollama initialise le contexte à 2 048 tokens par défaut, quelle que soit la capacité déclarée du modèle. Le prompt système seul consomme ~600 tokens, ce qui ne laissait aucune marge pour le contexte RAG.

**Pourquoi `CHUNK_SIZE=800` pour `mxbai-embed-large` ?**
Ce modèle d'embedding a une fenêtre maximale de 512 tokens (~2 000 caractères en anglais, moins en français). Des chunks de 1 500 chars la dépassaient régulièrement, provoquant des erreurs silencieuses d'indexation. 800 chars offre un bon compromis entre préservation du contexte et compatibilité.