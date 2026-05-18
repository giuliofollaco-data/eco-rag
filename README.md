# IPCC AR6 Chatbot - RAG avec Ollama & LangChain

Ce projet implémente un système de génération à enrichissement contextuel (RAG) permettant d'interroger localement le rapport de synthèse 2023 du GIEC (AR6). L'objectif est d'extraire des données climatiques précises tout en limitant les hallucinations du modèle grâce à un contexte sourcé.

## Problématique

Comment concevoir un système de RAG capable de garantir la fidélité factuelle et l'extraction de données complexes (graphiques, tableaux, chiffres, nuances de probabilité) à partir de documents volumineux et denses, tout en fonctionnant de manière souveraine et performante sur une machine locale ?

## Concept technique

Le système repose sur l'architecture RAG, qui combine la puissance de raisonnement des Large Language Models (LLM) avec la fiabilité de données externes spécifiques. Contrairement à un chatbot classique, ce moteur :
1. Indexe le rapport PDF du GIEC en fragments vectoriels (Embeddings).
2. Recherche les passages les plus pertinents pour chaque question utilisateur via une base de données vectorielle (ChromaDB).
3. Génère une réponse en français, strictement basée sur les extraits récupérés, garantissant ainsi la précision scientifique.

## Framework

* RTX 3070, 16GB RAM
* Orchestration : LangChain
* LLM Local : Ollama (Modèle llama3.2)
* Embeddings : mxbai-embed-large
* Base Vectorielle : ChromaDB
* Langage : Python 3.10+