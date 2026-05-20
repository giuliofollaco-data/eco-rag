import streamlit as st
from main import query_translation_chain, answer_chain
from vector import get_retrieved_context

st.set_page_config(page_title="Assistant GIEC AR6", page_icon="🌍")
st.title("🌍 Assistant expert GIEC AR6")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question sur le rapport..."):
    # Ajout au chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logique de traitement
    with st.chat_message("assistant"):
        with st.spinner("Analyse du rapport en cours..."):
            # 1. Traduction
            q_translated = query_translation_chain.invoke({"question": prompt}).strip()

            # 2. Retrieval
            context = get_retrieved_context(q_translated)

            # 3. Réponse
            response = answer_chain.invoke({"context": context, "question": prompt})

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
