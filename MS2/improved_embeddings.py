import streamlit as st
import time
import os
from neo4j import GraphDatabase, exceptions
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.language_models.llms import LLM 
from Create_kg import Neo4jConnection
from run_queries import execute_intent
from classify import build_intent_classifier, evaluate
from enteties import model, merge_entities, labels
import json
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

# Defining the 3 different llm models to be used 
MODEL_OPTIONS = {
    "Gemini 2.5 Flash": {"type": "gemini", "id": "gemini-2.5-flash"}, 
    "Mistral 7B Instruct": {"type": "hf", "id": "mistralai/Mistral-7B-Instruct-v0.2"},
    "Gemma 2B IT": {"type": "hf", "id": "google/gemma-2-2b-it"}
}
CONTEXT_OPTIONS = {
    "Cypher Query Only": "cypher_only",
    "Vector Store Only": "vector_only",
    "Both (Cypher + Vector)": "both"
}

class Neo4j:
    @staticmethod
    def read_config(filepath):
        config = {}
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    if not line.strip(): continue
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config[key.strip()] = value.strip().strip('"')
        except FileNotFoundError:
            print(f"Configuration file '{filepath}' not found.")
        except Exception as e:
            print(f"Error reading configuration file: {e}")
        return config

    def __init__(self, uri, user, password):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Connection Successful.")
        except exceptions.AuthError:
            print("Wrong username or password.")
            raise
        except exceptions.ServiceUnavailable:
            print("Connection failed - Database is likely down or unreachable.")
            raise
        except Exception as e:
            print(f"Error occurred: {e}")
            raise

    def CloseConnection(self):
        if self.driver:
            self.driver.close()
            print("Connection closed.")

    def fetch_journey_contexts(self):
        query = """
            MATCH (p:Passenger)-[:TOOK]->(j:Journey)
            MATCH (j)-[:ON]->(f:Flight)
            WITH DISTINCT j, p, f
            MATCH (f)-[:DEPARTS_FROM]->(dep:Airport)
            MATCH (f)-[:ARRIVES_AT]->(arr:Airport)
            RETURN 
                j.feedback_ID as id,
                j.food_satisfaction_score as food_satisfaction_score,
                j.arrival_delay_minutes as arrival_delay_minutes,
                j.actual_flown_miles as actual_flown_miles,
                j.number_of_legs as number_of_legs,
                p.loyalty_program_level as loyalty_program_level,
                p.generation as generation,
                f.flight_number as flight_number,
                f.fleet_type_description as fleet_type_description,
                min(dep.station_code) as origin,
                min(arr.station_code) as destination
            """
        documents = []

        try:
            with self.driver.session() as session:
                result = session.run(query)
                
                for record in result:
                    # Natural language description for embedding
                    text_description = (
                        f"A passenger from generation {record['generation']} with {record['loyalty_program_level']} loyalty program level "
                        f"flew from {record['origin']} to {record['destination']} on Flight {record['flight_number']}. "
                        f"The journey was performed on a {record['fleet_type_description']} aircraft, covering {record['actual_flown_miles']} miles "
                        f"across {record['number_of_legs']} leg(s). "
                        f"The flight arrived with a delay of {record['arrival_delay_minutes']} minutes. "
                        f"Regarding service, the passenger rated the food satisfaction {record['food_satisfaction_score']} out of 5."
                    )
                    
                    doc = Document(
                        page_content=text_description,
                        metadata={
                            "source_id": record['id'],
                            "properties": {
                                "loyalty_program_level": record['loyalty_program_level'],
                                "generation": record['generation'],
                                "flight_number": record['flight_number'],
                                "fleet_type_description": record['fleet_type_description'],
                                "number_of_legs": record['number_of_legs'],
                                "origin": record['origin'],
                                "destination": record['destination'],
                                "actual_flown_miles": record['actual_flown_miles'],
                                "arrival_delay_minutes": record['arrival_delay_minutes'],
                                "food_satisfaction_score": record['food_satisfaction_score']
                            }
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            print(f"Error fetching journey contexts: {e}")

        return documents

def extract_intent_and_entities(user_query):
    chain = build_intent_classifier()
    llm_response = evaluate(chain, user_query)
    llm_response = llm_response.strip()

    if llm_response.startswith("```"):
        llm_response = llm_response.strip("`").strip()

    if llm_response.startswith("json"):
        llm_response = llm_response[4:].strip()

    try:
        parsed = json.loads(llm_response)
    except json.JSONDecodeError:
        print("❌ LLM did NOT return valid JSON.")
        intent = "default"
        args = {}
    else:
        intent = parsed.get("intent", "default")
        args = parsed.get("arguments", {})
    
    print("\nDetected Intent:", intent)
    print("Arguments:", args)
    args = {}

    entities = model.predict_entities(user_query, labels)
    print(entities)
    entities = merge_entities(entities, user_query)
    print("\nExtracted Entities:")
    for ent in entities:
        print(ent["text"], "=>", ent["label"])

    config_neo = Neo4jConnection.read_config("config.txt")
    neo = Neo4jConnection(config_neo["NEO4J_URI"], config_neo["NEO4J_USER"], config_neo["NEO4J_PASSWORD"])

    print("\nRunning Query...")
    result_cypher = execute_intent(intent, args, entities, neo)
    print("\nQuery Result:", result_cypher)
    format_result = format_data_for_llm(result_cypher, intent)
    print("\nFormatted Result:", format_result)

    neo.CloseConnection()
    return format_result, intent
    
def create_embeddings_mini(documents, save_path="journey_vector_store"):
    if not documents:
        print("No documents provided to embed.")
        return None

    try:
        print("Initializing MiniLM embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print(f"Generating vectors for {len(documents)} documents...")
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model)
        if save_path:
            vectorstore.save_local(save_path)
            print(f"Success! Vector store saved to '{save_path}' folder.")
            
        return vectorstore

    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

# NEW FUNCTION: Convert retrieved documents to structured format
def format_retrieved_docs_to_structured(documents, add_prefix=True):
    """
    Converts retrieved vector store documents to the same structured format
    as Cypher query results for unified LLM processing.
    
    Args:
        documents: List of LangChain Documents with metadata
        add_prefix: Whether to add an informative prefix (default: True)
        
    Returns:
        str: Formatted string matching Cypher query output format
    """
    if not documents:
        return "No contextual journey records were retrieved from the vector store."
    
    formatted_records = []
    
    for doc in documents:
        properties = doc.metadata.get('properties', {})
        if not properties:
            continue
            
        # Create key:value pairs matching Cypher format
        key_value_pairs = [f"{key}:{value}" for key, value in properties.items()]
        record_string = ", ".join(key_value_pairs)
        formatted_records.append(record_string)
    
    if not formatted_records:
        return "No valid structured records found in the retrieved documents."
    
    # Join records with semicolon, matching Cypher format
    structured_data = "; ".join(formatted_records) + "."
    
    # Add informative prefix (similar to Cypher format)
    if add_prefix:
        num_records = len(formatted_records)
        return f"Retrieved {num_records} relevant journey record(s) from semantic search: {structured_data}"
    
    return structured_data

@st.cache_resource
def setup_static_rag_components(config_path='config.txt', vector_store_path="journey_vector_store"):
    st.info("Starting static RAG component setup... (Loading data and embeddings)")

    config = Neo4j.read_config(config_path)
    if not config: 
        return None, None
    
    uri = config.get('NEO4J_URI')
    user = config.get('NEO4J_USER')
    password = config.get('NEO4J_PASSWORD')
    
    if not all([uri, user, password]):
        st.error("Missing Neo4j credentials in config.txt.")
        return None, None
    
    importer = None
    try:
        importer = Neo4j(uri, user, password)
        docs = importer.fetch_journey_contexts()
        st.success(f"Successfully fetched {len(docs)} documents from Neo4j.")

        vector_store = create_embeddings_mini(docs, save_path=vector_store_path)
        return config, vector_store
    
    except Exception as e:
        st.error(f"Failed to set up static components: {e}")
        return None, None
    finally:
        if importer:
            importer.CloseConnection()

def initialize_rag_chain(config, vector_store, selected_model_data: dict, context_mode: str):
    """Initializes the correct LLM and RetrievalQA chain for the selected model and context mode."""

    # UPDATED PROMPTS - Now all use structured data format
    if context_mode == "cypher_only":
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the structured query results provided below.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Base your answer ONLY on the query results provided. Do NOT use general knowledge.
        3. If the query results are empty or don't contain the answer, state: "The specific data needed to answer this question could not be found in the query results."
        4. Keep your answer concise and focused on the key metrics.
        5. Do NOT mention technical terms like "Cypher", "Neo4j", "intent classification", or "database queries" in your final answer.
        
        # DATA SOURCE
        {cypher_context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["input", "cypher_context"]
        
    elif context_mode == "vector_only":
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the structured journey records retrieved from the knowledge base.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Base your answer ONLY on the structured records provided below. Do NOT use general knowledge.
        3. If the records do not contain the answer, state: "The specific data needed to answer this question could not be found in the available journey records."
        4. Keep your answer concise and focused on the key metrics (delay, miles, satisfaction, loyalty, generation, routes).
        5. Do NOT mention "vector store", "semantic search", "embeddings", or technical implementation details in your final answer.
        
        # DATA SOURCE
        {vector_context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["vector_context", "input"]
        
    else:  # both
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions using BOTH data sources provided below in structured format.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Both data sources use the same structured format for easy comparison.
        3. The first data source (targeted query) is specifically filtered for this question - prioritize it.
        4. The second data source (semantic search) provides broader contextual journey records - use it for additional validation and context.
        5. If you see duplicate records between sources, mention the information only once in your answer.
        6. If neither source contains the answer, state: "The specific data needed to answer this question could not be found in the system's records."
        7. Keep your answer concise and focused on the key metrics.
        8. Do NOT mention "vector store", "embeddings", "Cypher", "intent classification", or technical implementation details in your final answer.
        
        # DATA SOURCE 1 (Targeted Query Results)
        {cypher_context}
        
        # DATA SOURCE 2 (Contextual Journey Records)
        {vector_context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["vector_context", "input", "cypher_context"]
    
    qa_prompt = PromptTemplate(
        template=prompt,
        input_variables=input_variables
    )

    retriever = None
    if context_mode in ["vector_only", "both"]:
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    model_type = selected_model_data["type"]
    model_id = selected_model_data["id"]
    llm_wrapper = None

    if model_type == "gemini":
        google_key = config.get('GOOGLE_API_KEY')
        if not google_key:
            st.error("GOOGLE_API_KEY missing in config.txt for Gemini model.")
            return None, None, None
        os.environ["GOOGLE_API_KEY"] = google_key
        
        llm_wrapper = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.1,
            max_tokens=500,
        )
    elif model_type == "hf":
        hf_token = config.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            st.error("HUGGINGFACE_TOKEN missing in config.txt for Hugging Face model.")
            return None, None, None
        
        hf_llm_worker = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.1,
            max_new_tokens=500,
        )
        llm_wrapper = ChatHuggingFace(llm=hf_llm_worker)
    else:
        st.error(f"Unknown model type: {model_type}")
        return None, None, None
        
    if llm_wrapper is None: 
        return None, None, None
    
    return llm_wrapper, retriever, qa_prompt

def format_data_for_llm(data_list, intent):
    """
    Transforms a list of dictionaries into a single string of key:value pairs
    separated by commas, with each dictionary's output separated by a semicolon.
    """
    if not data_list:
        return f"No results found for the query with intent '{intent}'."
    
    if isinstance(data_list, dict) and 'error' in data_list:
        return f"Query execution failed for intent '{intent}': {data_list['error']}"
    
    if not isinstance(data_list, list):
        return f"Invalid query result format for intent '{intent}'."
    
    formatted_dicts = []
    
    for data_dict in data_list:
        if not isinstance(data_dict, dict):
            continue
            
        key_value_pairs = [f"{key}:{value}" for key, value in data_dict.items()]
        dict_string = ", ".join(key_value_pairs)
        formatted_dicts.append(dict_string)
    
    if not formatted_dicts:
        return f"No valid results found for the query with intent '{intent}'."
        
    final_string = "; ".join(formatted_dicts) + "."
    return f"The results of the query with intent '{intent}' are: {final_string}"

def main():
    st.set_page_config(page_title="✈️ Multi-LLM Neo4j RAG Chatbot")
    st.title("✈️ Multi-LLM Neo4j RAG Chatbot")
    st.markdown("Query customer journey data from Neo4j using different Large Language Models.")

    config, vectorstore = setup_static_rag_components()
    
    if config is None or vectorstore is None:
        st.error("Fatal error during application startup. Please check logs and config.txt.")
        st.stop()

    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_name = st.selectbox(
            "Choose your LLM:",
            options=list(MODEL_OPTIONS.keys()),
            index=0
        )
    
    with col2:
        selected_context_name = st.selectbox(
            "Choose Context Source:",
            options=list(CONTEXT_OPTIONS.keys()),
            index=2,
            help="Select which data source to use for answering questions:\n"
                 "- Cypher Query Only: Uses only targeted database queries\n"
                 "- Vector Store Only: Uses only semantic search on journey samples\n"
                 "- Both: Combines both approaches for comprehensive answers"
        )
    
    selected_model_data = MODEL_OPTIONS[selected_model_name]
    selected_context_mode = CONTEXT_OPTIONS[selected_context_name]
    
    with st.spinner(f"Initializing {selected_model_name} chain components..."):
        llm, retriever, qa_prompt = initialize_rag_chain(config, vectorstore, selected_model_data, selected_context_mode)

    if llm is None:
        st.error(f"Failed to initialize LLM components for {selected_model_name}. Check the required token in config.txt.")
        st.stop()
    
    st.success(f"RAG Chatbot is online! Using **{selected_model_name}** (`{selected_model_data['id']}`) with **{selected_context_name}** context.")

    user_query = st.text_input("Your Question:", placeholder="Which generation of passengers reported the lowest food satisfaction?")

    if user_query:
        format_result, intent = extract_intent_and_entities(user_query)

        with st.spinner(f"Generating answer using {selected_model_name}..."):
            try:
                if selected_context_mode == "cypher_only":
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": user_query,
                        "cypher_context": format_result
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    source_documents = []
                    
                elif selected_context_mode == "vector_only":
                    # Retrieve documents and convert to structured format
                    retrieved_docs = retriever.invoke(user_query)
                    vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": user_query,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    source_documents = retrieved_docs
                    
                else:  # both
                    # Retrieve documents and convert to structured format
                    retrieved_docs = retriever.invoke(user_query)
                    vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": user_query,
                        "cypher_context": format_result,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    source_documents = retrieved_docs
                    
            except Exception as e:
                st.error(f"An error occurred during chain execution: {e}")
                answer = "Error generating response."
                source_documents = []

        st.subheader("Generated Answer")
        
        source_citation_text = ""
        is_cypher_result_meaningful = format_result and not format_result.endswith("The results of the query with intent 'default' are: .")
        
        if is_cypher_result_meaningful:
            source_citation_text += "The answer was grounded using results from a specific Cypher query"
            if source_documents:
                source_citation_text += " and structured journey records from the vector store."
            else:
                source_citation_text += "."
        elif source_documents:
            source_citation_text += "The answer was grounded using structured journey records from the vector store."
            
        if source_citation_text:
            st.caption(f"**Source Grounding:** {source_citation_text}")
        
        st.info(answer)

        if is_cypher_result_meaningful:
            st.subheader("💡 Cypher Query Result (Primary Grounding Data)")
            st.markdown(f"The intent was classified as `{intent}`, and the resulting Neo4j query provided the following structured data:")
            st.code(format_result, language='text')

        if source_documents:
            st.subheader("📚 Vector Store Retrieved Records (Secondary Grounding Data)")
            st.markdown(f"**{len(source_documents)} journey record(s)** were retrieved using semantic similarity search.")
            
            # Show what was actually passed to the LLM (structured format)
            vector_structured = format_retrieved_docs_to_structured(source_documents)
            
            st.markdown("#### 🤖 Structured Data Passed to LLM:")
            st.info("This is the formatted structured data that was provided to the LLM for answer generation:")
            st.code(vector_structured, language='text')
            
            # Show what the embeddings actually matched on (natural language)
            st.markdown("#### 🔍 Natural Language Text Matched by Embeddings:")
            st.info("These are the semantic descriptions that the embedding model found most similar to your query:")
            
            with st.expander("View All Matched Semantic Descriptions", expanded=True):
                for i, doc in enumerate(source_documents, 1):
                    source_id = doc.metadata.get('source_id', 'N/A')
                    st.markdown(f"**Match {i}** (Journey ID: `{source_id}`)")
                    # This is the natural language description used for embedding
                    st.markdown(f"> {doc.page_content}")
                    st.markdown("")  # Add spacing
            
            # Optionally show the raw structured data for each record
            with st.expander("View Individual Structured Records (JSON Format)"):
                for i, doc in enumerate(source_documents, 1):
                    source_id = doc.metadata.get('source_id', 'N/A')
                    st.markdown(f"---")
                    st.markdown(f"**Record {i}:** (Journey ID: `{source_id}`)")
                    
                    record_data = doc.metadata.get('properties', {})
                    if record_data:
                        st.json(record_data)
                    else:
                        st.markdown("*No structured data available*")


if __name__ == "__main__":
    main()