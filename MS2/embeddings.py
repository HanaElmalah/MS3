
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

PREDEFINED_QUESTIONS = [
    "",
    "What are the top 5 routes by flight count?",
    "Which generations flew the most ?",
    "What's the route that had the most delay ?",
    "what is the flight that had the route from LAX to IAX?"
    # "what are all the diff genrations ?",
    # "what are the different loyalty ?",
]

class Neo4j:
# Method to read the configuration file
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
# Method to establish a connection to the Neo4j database
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
# Method to close the connection to the Neo4j database
    def CloseConnection(self):
        if self.driver:
            self.driver.close()
            print("Connection closed.")
# Method to fetch the journey contexts from the Neo4j database which include every journey connected to a passenger who was on a certain flight from a certain airport to another airport
# A feature text is then formed using a specified context for every journey, which includes the flight details, passenger details, and service details
# All feature texts are collected in documents array which is then passed to the create_embeddings method to generate embeddings for each document and create a FAISS index for the embeddings
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
                min(dep.station_code) as destination,
                min(arr.station_code) as origin
            """
        documents = []

        try:
            with self.driver.session() as session:
                result = session.run(query)
                
                for record in result:
                    # text_description = (
                    #     f"Loyalty Level: {record['loyalty']}. "
                    #     f"Generation: {record['gen']}. "
                    #     f"Flight Number: {record['flight_num']}. "
                    #     f"Fleet Type: {record['fleet']}. "
                    #     f"Number of Legs: {record['legs']}. "  
                    #     f"Origin Airport: {record['origin']}. "
                    #     f"Destination Airport: {record['dest']}. "
                    #     f"Flight Distance: {record['miles']} miles. "
                    #     f"Arrival Delay: {record['delay']} minutes. "
                    #     f"Food Satisfaction Score: {record['food']}/5."
                    # )
                    # text_description = (
                    #     f"loyalty_program_level: {record['loyalty']}. "
                    #     f"generation: {record['gen']}. "
                    #     f"flight_number: {record['flight_num']}. "
                    #     f"fleet_type_description: {record['fleet']}. "
                    #     f"number_of_legs: {record['legs']}. "  
                    #     f"Origin Airport: {record['origin']}. "
                    #     f"destination: {record['dest']}. "
                    #     f"actual_flown_miles: {record['miles']} miles. "
                    #     f"arrival_delay_minutes: {record['delay']} minutes. "
                    #     f"food_satisfaction_score: {record['food']}/5."
                    # )
                    text_description = (
                        f"A passenger from generation{record['generation']} with {record['loyalty_program_level']} loyalty program level "
                        f"flew from {record['origin']} to {record['destination']} on Flight {record['flight_number']}. "
                        f"The journey was performed on a {record['fleet_type_description']} aircraft, covering {record['actual_flown_miles']} miles "
                        f"across {record['number_of_legs']} leg(s). "
                        f"The flight arrived with a delay of {record['arrival_delay_minutes']} minutes. "
                        f"Regarding service, the passenger rated the food satisfaction {record['food_satisfaction_score']} out of 5."
                    )
                    
                    doc = Document(
                        page_content=text_description,
                        metadata={"source_id": record['id'],
                                    "properties": record
                                    }
                    )
                    #print(doc.metadata['properties'])
                    documents.append(doc)
                    
                    
                    
        except Exception as e:
            print(f"Error fetching journey contexts: {e}")

        return documents

def extract_intent_and_entities(user_query):
    # 5️⃣ Create and Invoke the LCEL Chain
        # This part requires the imports: create_stuff_documents_chain and create_retrieval_chain
        chain = build_intent_classifier()
        llm_response = evaluate(chain, user_query)
        llm_response = llm_response.strip()

        # Remove surrounding ``` blocks
        if llm_response.startswith("```"):
            llm_response = llm_response.strip("`").strip()

        # Remove leading "json"
        if llm_response.startswith("json"):
            llm_response = llm_response[4:].strip()

        try:
            parsed = json.loads(llm_response)
        except json.JSONDecodeError:
            print("❌ LLM did NOT return valid JSON.")
            # Fallback for error case
        intent = "default"
        args = {}
        # NOTE: You might want to handle this better in a real app
        
        intent = parsed.get("intent", "default")
        args = parsed.get("arguments", {})
        print("\nDetected Intent:", intent)
        print("Arguments:", args)
        args={} # Existing line from original code

        # 2️⃣ Extract entities
        entities = model.predict_entities(user_query, labels)
        print(entities)
        entities = merge_entities(entities ,user_query)
        print("\nExtracted Entities:")
        for ent in entities:
            print(ent["text"], "=>", ent["label"])

        # 3️⃣ Connect to Neo4j
        config_neo = Neo4jConnection.read_config("config.txt")
        neo = Neo4jConnection(config_neo["NEO4J_URI"], config_neo["NEO4J_USER"], config_neo["NEO4J_PASSWORD"])

        # 4️⃣ Execute intent
        print("\nRunning Query...")
        result_cypher , query_itself = execute_intent(intent, args, entities, neo)
        print("\nQuery Result:", result_cypher)
        # This is the formatted output to be injected into the LLM context
        format_result = format_data_for_llm(result_cypher,intent)
        print("\nFormatted Result:", format_result)

        neo.CloseConnection()
        return format_result, intent , query_itself
    
def create_embeddings_mini(documents, save_path="journey_vector_store"):
    """
    Takes a list of LangChain Documents, generates embeddings using MiniLM,
    creates a FAISS index, and saves it to disk.
    """
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
    
def create_embeddings_base(documents, save_path="journey_vector_store"):
    """
    Takes a list of LangChain Documents, generates embeddings using MiniLM,
    creates a FAISS index, and saves it to disk.
    """
    if not documents:
        print("No documents provided to embed.")
        return None

    try:
        print("Initializing MiniLM embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
    
def create_embeddings_baai(documents, save_path="journey_vector_store"):
    """
    Takes a list of LangChain Documents, generates embeddings using MiniLM,
    creates a FAISS index, and saves it to disk.
    """
    if not documents:
        print("No documents provided to embed.")
        return None

    try:
        print("Initializing MiniLM embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
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
    
# def format_output(result):
#     """
#     Extracts all key properties from the retrieved Neo4j Record metadata 
#     and formats them into a single-line, dot-separated 'Attribute: Value.' string 
#     for the LLM context.
#     """
#     formatted_context_list = []

#     for i, doc in enumerate(result.get_source_documents,1):
#         record = doc.metadata['properties']        
#         # We must map the attribute names used in the original document creation 
#         # (e.g., 'loyalty_program_level') to the simplified names in your format (e.g., 'loyalty').
        
#         # 1. Define the facts to include and map them to their clean attribute names
#         # We use .get() for safety against missing keys
#         facts_to_include = {
#             "loyalty_program_level": record.get('loyalty_program_level'),
#             "generation": record.get('generation'),
#             "flight_number": record.get('flight_number'),
#             "fleet_type_description": record.get('fleet_type_description'),
#             "number_of_legs": record.get('number_of_legs'),
#             "origin": record.get('origin'),
#             "destination": record.get('destination'),
#             "actual_flown_miles": f"{record.get('actual_flown_miles')} miles",
#             "arrival_delay_minutes": f"{record.get('arrival_delay_minutes')} minutes",
#             "food_satisfaction_score": record.get('food_satisfaction_score'),
#         }

#         # 2. Construct the final unified string by joining the facts
#         formatted_item = ""
#         for attribute, value in facts_to_include.items():
#             if value is not None:
#                 # Add a space before the attribute for cleaner separation when concatenating
#                 formatted_item += f" {attribute}: {value}."
        
#         # Remove any leading space that results from the initial concatenation
#         formatted_item = formatted_item.strip()
        
#         if formatted_item:
#             formatted_context_list.append(formatted_item)

#     # 3. Return the list joined by newlines for the LLM context
#     return "\n".join(formatted_context_list)


@st.cache_resource
def setup_static_rag_components(config_path= 'config.txt',vector_store_path="journey_vector_store"):
    """
    Initializes and caches the static components (config, vector store).
    """
    st.info("Starting static RAG component setup... (Loading data and embeddings)")

    # Loading config data to initialize connetion to Neo4j
    config = Neo4j.read_config(config_path)
    if not config: return None, None
    
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

        vector_store= create_embeddings_mini(docs, save_path=vector_store_path)
        return config,vector_store
    
    except Exception as e:
        st.error(f"Failed to set up static components: {e}")
        return None, None
    finally:
        if importer:
            importer.CloseConnection()


# def initialize_rag_chain(config,vector_store, selected_model_data:dict):
#     """Initializes the correct LLM and RetrievalQA chain for the selected model."""

# # Create the prompt template
#     # prompt = """
#     # You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the customer journey data provided below.

#     # # IMPORTANT RULES
#     # 1. Maintain a professional, objective, and data-driven tone.
#     # 2. If the retrieved context does not contain the answer, you MUST state, "The specific data needed to answer this question could not be found in the system's records." Do NOT guess or use general knowledge.
#     # 3. Keep your answer concise and focused on the key metrics (delay, miles, satisfaction, loyalty, generation).
#     # 4. Do NOT mention "vector store", "embeddings", or "LangChain" in your final answer.

#     # # CONTEXT
#     # {context}

#     # # QUESTION
#     # {question}

#     # # ANALYST ANSWER:
#     # """
#     # qa_prompt = PromptTemplate(
#     #     template = prompt,
#     #     input_variables = ["context", "question"]
#     # )
#     prompt = """
#     You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the customer journey data provided below.

#     # IMPORTANT RULES
#     1. Maintain a professional, objective, and data-driven tone.
#     2. If the retrieved context does not contain the answer, you MUST state, "The specific data needed to answer this question could not be found in the system's records." Do NOT guess or use general knowledge.
#     3. Keep your answer concise and focused on the key metrics (delay, miles, satisfaction, loyalty, generation).
#     4. Do NOT mention "vector store", "embeddings", or "LangChain" in your final answer.
    
#     # CYPER QUERY RESULTS
#     {cypher_context} 
    
#     # CONTEXT (Retrieved Journey Samples)
#     {context}

#     # QUESTION
#     {input}

#     # ANALYST ANSWER:
#     """
#     qa_prompt = PromptTemplate(
#         template = prompt,
#         # **IMPORTANT CHANGE**: Add 'cypher_context' to input_variables
#         input_variables = ["context", "input", "cypher_context"] 
#     )
# # Create the retriever
#     retriever = vector_store.as_retriever(search_kwargs={"k": 15})
#     #print("ret_raw", retriever_raw)
#     #retriever = format_output(retriever_raw)

# # Create the LLM
#     model_type = selected_model_data["type"]
#     model_id = selected_model_data["id"]
#     llm_wrapper = None

#     if model_type == "gemini":
#         google_key = config.get('GOOGLE_API_KEY')
#         if not google_key:
#             st.error("GOOGLE_API_KEY missing in config.txt for Gemini model.")
#             return None
#         os.environ["GOOGLE_API_KEY"] = google_key
        
#         llm_wrapper = ChatGoogleGenerativeAI(
#             model=model_id,
#             temperature=0.1,
#             max_new_tokens=500,
#         )
#     elif model_type == "hf":
#         hf_token = config.get('HUGGINGFACE_TOKEN')
#         if not hf_token:
#             st.error("HUGGINGFACE_TOKEN missing in config.txt for Hugging Face model.")
#             return None
        
#         hf_llm_worker = HuggingFaceEndpoint(
#             repo_id=model_id,
#             huggingfacehub_api_token=hf_token,
#             temperature=0.1,
#             max_new_tokens=500,
#             # We must NOT set 'task' here, let the ChatHuggingFace wrapper handle the formatting
#             # This combination prevents both the "Bad Request" and "llm Field required" errors.
#         )
#         llm_wrapper = ChatHuggingFace(llm=hf_llm_worker)
#     else:
#         st.error(f"Unknown model type: {model_type}")
#         return None
#     if llm_wrapper is None: return None
    
#     # qa_chain = RetrievalQA.from_chain_type(
#     #     llm=llm_wrapper,
#     #     retriever= retriever,
#     #     chain_type="stuff",
#     #     return_source_documents=True,
#     #     chain_type_kwargs={"prompt": qa_prompt}
    
#     # )
#     # return qa_chain
#     return llm_wrapper, retriever, qa_prompt

def initialize_rag_chain(config, vector_store, selected_model_data: dict, context_mode: str):
    """Initializes the correct LLM and RetrievalQA chain for the selected model and context mode."""

    # Create different prompts based on context mode
    if context_mode == "cypher_only":
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the Cypher query results provided below.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Base your answer ONLY on the Cypher query results. Do NOT use general knowledge.
        3. If the query results are empty or don't contain the answer, state: "The specific data needed to answer this question could not be found in the query results."
        4. Keep your answer concise and focused on the key metrics.
        5. Do NOT mention technical terms like "Cypher", "Neo4j", or "database queries" in your final answer.
        
        # CYPHER QUERY RESULTS
        {cypher_context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["input", "cypher_context"]
        
    elif context_mode == "vector_only":
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions based ONLY on the customer journey samples retrieved from the knowledge base.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Base your answer ONLY on the retrieved journey samples below. Do NOT use general knowledge.
        3. If the retrieved context does not contain the answer, state: "The specific data needed to answer this question could not be found in the available journey records."
        4. Keep your answer concise and focused on the key metrics (delay, miles, satisfaction, loyalty, generation, routes).
        5. Do NOT mention "vector store", "embeddings", or technical implementation details in your final answer.
        6 DO NOT mention any extra attributes that are not mentioned in the prompt.
        
        # CONTEXT (Retrieved Journey Samples)
        {context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["context", "input"]
        
    else:  # both
        prompt = """
        You are a highly skilled **Airline Customer Analyst**. Your goal is to accurately answer airline company employee questions using BOTH the specific query results AND contextual journey samples provided below.

        # IMPORTANT RULES
        1. Maintain a professional, objective, and data-driven tone.
        2. Prioritize the Cypher query results as they are specifically targeted to the question. 
        3.Use the journey samples for additional context and validation.
        4. If neither source contains the answer, state: "The specific data needed to answer this question could not be found in the system's records."
        5. Keep your answer concise and focused on the key metrics (delay, miles, satisfaction, loyalty, generation).
        6. Do NOT mention "vector store", "embeddings", "Cypher", or technical implementation details in your final answer.
        7. If the cypher query results have duplicate values to the journey samples, output only one of the two duplicated values.
        8. DO NOT mention any extra attributes that are not mentioned in the prompt.
        
        # CYPHER QUERY RESULTS (Primary Data)
        {cypher_context}
        
        # CONTEXT (Retrieved Journey Samples for Additional Context)
        {context}

        # QUESTION
        {input}

        # ANALYST ANSWER:
        """
        input_variables = ["context", "input", "cypher_context"]
    
    qa_prompt = PromptTemplate(
        template=prompt,
        input_variables=input_variables
    )

    # Create the retriever (only needed for vector_only and both modes)
    retriever = None
    if context_mode in ["vector_only", "both"]:
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    # Create the LLM
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

# def format_data_for_llm(data_list, intent):
#     """
#     Transforms a list of dictionaries into a single string of key:value pairs
#     separated by commas, with each dictionary's output separated by a semicolon.
#     The string ends with a period.

#     Args:
#         data_list (list): A list of dictionaries to process.

#     Returns:
#         str: The formatted string.
#     """
#     formatted_dicts = []
    
#     # 1. Iterate through each dictionary in the list
#     for data_dict in data_list:
#         # 2. Convert key:value pairs into 'key:value' strings
#         key_value_pairs = [f"{key}:{value}" for key, value in data_dict.items()]
        
#         # 3. Join the pairs with a comma and space
#         dict_string = ", ".join(key_value_pairs)
        
#         formatted_dicts.append(dict_string)
        
#     # 4. Join the formatted dictionary strings with a semicolon and space
#     # 5. Append a period to the very end
#     final_string = "; ".join(formatted_dicts) + "."
#     final_string2 = f"The results of the query with intent '{intent}' are: {final_string}"
    
# return final_string2
def format_data_for_llm(data_list, intent):
    """
    Transforms a list of dictionaries into a single string of key:value pairs
    separated by commas, with each dictionary's output separated by a semicolon.
    The string ends with a period.

    Args:
        data_list (list): A list of dictionaries to process.
        intent (str): The detected intent.

    Returns:
        str: The formatted string.
    """
    # Handle error cases
    if not data_list:
        return f"No results found for the query with intent '{intent}'."
    
    # Check if data_list is a dict with an error key
    if isinstance(data_list, dict) and 'error' in data_list:
        return f"Query execution failed for intent '{intent}': {data_list['error']}"
    
    # Ensure data_list is actually a list
    if not isinstance(data_list, list):
        return f"Invalid query result format for intent '{intent}'."
    
    formatted_dicts = []
    
    # 1. Iterate through each dictionary in the list
    for data_dict in data_list:
        # Skip if not a dictionary
        if not isinstance(data_dict, dict):
            continue
            
        # 2. Convert key:value pairs into 'key:value' strings
        key_value_pairs = [f"{key}:{value}" for key, value in data_dict.items()]
        
        # 3. Join the pairs with a comma and space
        dict_string = ", ".join(key_value_pairs)
        
        formatted_dicts.append(dict_string)
    
    # If no valid data was processed
    if not formatted_dicts:
        return f"No valid results found for the query with intent '{intent}'."
        
    # 4. Join the formatted dictionary strings with a semicolon and space
    # 5. Append a period to the very end
    final_string = "; ".join(formatted_dicts) + "."
    final_string2 = f"The results of the query with intent '{intent}' are: {final_string}"
    
    return final_string2




def main():
    st.set_page_config(page_title="✈️ Multi-LLM Neo4j RAG Chatbot")
    st.title("✈️ Multi-LLM Neo4j RAG Chatbot")
    st.markdown("Query customer journey data from Neo4j using different Large Language Models.")

    # 1. Setup static components (config and vectorstore)
    config, vectorstore = setup_static_rag_components()
    
    if config is None or vectorstore is None:
        st.error("Fatal error during application startup. Please check logs and config.txt.")
        st.stop()

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar: Model and Context Selection ---
    st.sidebar.title("Optional Panel")
    st.sidebar.markdown("_ _ _ _")
    
    selected_model_name = st.sidebar.selectbox(
        "Choose your LLM:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )

    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("_ _ _ _")
    
    selected_context_name = st.sidebar.selectbox(
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

    # --- Initialize the QA Chain ---
    with st.spinner(f"Initializing {selected_model_name} chain components..."):
        llm, retriever, qa_prompt = initialize_rag_chain(config, vectorstore, selected_model_data, selected_context_mode)
    
    if llm is None:
        st.error(f"Failed to initialize LLM components for {selected_model_name}. Check the required token in config.txt.")
        st.stop()
    
    st.success(f"RAG Chatbot is online! Using **{selected_model_name}** (`{selected_model_data['id']}`).")

    # --- Chat Input ---
    # Use a selectbox for pre-selection
    selected_query = st.selectbox(
        "Choose a sample question or write your own below:",
        options=PREDEFINED_QUESTIONS,
        index=0, # Default to the first option
    )

    # st.chat_input provides the "write" option
    user_query = st.chat_input("Your Question:"
    )

    # Determine the final query to process
    final_query = None

    if user_query:
        final_query = user_query
    elif selected_query:
        final_query = selected_query

    if final_query:
        user_query = final_query
        # --- Process user query ---
        format_result, intent , query_itself = extract_intent_and_entities(user_query)

        # Generate bot answer
        with st.spinner(f"Generating answer using {selected_model_name}..."):
            try:
                # --- Start Timer ---
                start_time = time.time()
                if selected_context_mode == "cypher_only":
                    chain = qa_prompt | llm
                    result = chain.invoke({"input": user_query, "cypher_context": format_result})
                    answer = result.content if hasattr(result, "content") else str(result)
                    source_documents = []
                elif selected_context_mode == "vector_only":
                    document_chain = create_stuff_documents_chain(llm, qa_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    result = retrieval_chain.invoke({"input": user_query})
                    answer = result.get("answer", "No answer generated.")
                    source_documents = result.get("context", [])
                else:  # both
                    document_chain = create_stuff_documents_chain(llm, qa_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    result = retrieval_chain.invoke({"input": user_query, "cypher_context": format_result})
                    answer = result.get("answer", "No answer generated.")
                    source_documents = result.get("context", [])
                
                # --- End Timer and Calculate Latency ---
                end_time = time.time()
                latency = end_time - start_time
            except Exception as e:
                answer = f"Error generating response: {e}"
                source_documents = []
                latency = 0.0 # Assign 0 or a very high value on error

        # Save user and assistant messages with sources in session state
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "format_result": format_result,
            "intent": intent,
            "source_documents": source_documents,
            "query_itself": query_itself,
            # --- ADD NEW METRICS HERE ---
            "latency": latency,
            "answer_length": len(answer), # Use char count as a proxy for token count
            "model_name": selected_model_name,
            "context_mode": selected_context_name,
        })

    # --- Display chat history ---
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

        # Only display sources for assistant messages
        if msg["role"] == "assistant":
            format_result = msg.get("format_result", "")
            intent = msg.get("intent", "")
            source_documents = msg.get("source_documents", [])
            query_itself = msg.get("query_itself", "N/A")

            # Use an expander to hide/show source info
            if format_result or source_documents:
                with st.expander("Show Context & Cypher Results"):
                    # Cypher Query Result
                    is_cypher_result_meaningful = format_result and not format_result.endswith("The results of the query with intent 'default' are: .")
                    if is_cypher_result_meaningful:
                        st.subheader("💡 Cypher Query Result (Primary Grounding Data)")
                        st.markdown(f"The intent was classified as `{intent}`, and the resulting Neo4j query provided the following specific data:")
                        st.code(format_result, language='text')

                    # Add the raw Cypher query
                    if query_itself and query_itself != "N/A":
                        st.subheader("💻 Executed Cypher Query")
                        st.markdown(f"The exact Cypher query executed against the Neo4j database was:")
                        st.code(query_itself, language='cypher') # <-- ADD THIS BLOCK

                    # Vector Store Context
                    if source_documents:
                        st.subheader("📚 KG-Retrieved Context (Vector Store Data)")
                        st.markdown("The following records were retrieved from the vector store and provided to the LLM as additional context:")
                        for i, doc in enumerate(source_documents, 1):
                            source_id = doc.metadata.get('source_id', 'N/A')
                            st.markdown(f"---")
                            st.markdown(f"**Context Document {i}:** (Journey ID: `{source_id}`)")
                            st.markdown("**A. Semantic Text Feature (Used for Embedding Retrieval):**")
                            st.code(doc.page_content, language='text')

                            record_data = doc.metadata.get('properties', {})
                            if record_data:
                                st.markdown("**B. Structured Data Record (Used for LLM Grounding):**")
                                st.json(record_data)
                            else:
                                st.markdown("**B. Structured Data Record:** N/A")


if __name__ == "__main__":
    main()


