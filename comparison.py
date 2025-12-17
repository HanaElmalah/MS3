import streamlit as st
import time
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# Import your existing components
from improved_embeddings import (
    setup_static_rag_components,
    initialize_rag_chain,
    extract_intent_and_entities,
    format_retrieved_docs_to_structured,
    MODEL_OPTIONS,
    CONTEXT_OPTIONS
)

# # --- Define the specific exception we want to catch (Placeholder) ---
# # When using Google GenAI, this is a common specific error.
# # You might need to adjust this based on the specific LLM provider you are using.
# try:
#     from google.api_core.exceptions import ResourceExhausted, InvalidArgument
# except ImportError:
#     # Fallback for models without specific client libraries
#     class ResourceExhausted(Exception): pass
#     class InvalidArgument(Exception): pass
# # ---------------------------------------------------------------------

# class ModelComparison:
# # ... (ModelComparison class methods remain the same) ...
#     """Handles comparison metrics and evaluation for multiple LLM models"""
    
#     def __init__(self):
#         self.results = []
#         self.ground_truth = self.load_ground_truth()
    
#     def load_ground_truth(self):
#         """Define ground truth answers for evaluation questions"""
#         return {
#             "Which generation of passengers reported the lowest food satisfaction?": {
#                 "expected_answer": "Millennials",
#                 "expected_entities": ["Millennials", "generation"],
#                 "query_type": "aggregation"
#             },
#             "What is the average food satisfaction score for flights from CAI to HBE?": {
#                 "expected_answer": "specific numeric value",
#                 "expected_entities": ["CAI", "HBE", "food satisfaction"],
#                 "query_type": "route_specific"
#             },
#             "Show me the top 3 routes by passenger satisfaction": {
#                 "expected_answer": "list of routes with satisfaction scores",
#                 "expected_entities": ["routes", "satisfaction", "top"],
#                 "query_type": "ranking"
#             },
#             "How many passengers from the Baby Boomer generation flew on route JFK to LAX?": {
#                 "expected_answer": "specific count",
#                 "expected_entities": ["Baby Boomer", "JFK", "LAX"],
#                 "query_type": "count"
#             },
#             "What routes have the highest average delay?": {
#                 "expected_answer": "routes with delay information",
#                 "expected_entities": ["routes", "delay", "highest"],
#                 "query_type": "ranking"
#             }
#         }
    
#     def evaluate_response(self, question, answer, model_name, context_mode, 
#                           response_time, cypher_result, vector_docs):
#         """Comprehensive evaluation of a model's response"""
        
#         metrics = {
#             "model": model_name,
#             "context_mode": context_mode,
#             "question": question,
#             "answer": answer,
#             "timestamp": datetime.now().isoformat(),
            
#             # Quantitative Metrics
#             "response_time_seconds": response_time,
#             "answer_length_chars": len(answer),
#             "answer_length_words": len(answer.split()),
            
#             # Context Usage Metrics
#             "used_cypher": bool(cypher_result and "No results found" not in cypher_result),
#             "used_vector": bool(vector_docs and len(vector_docs) > 0),
#             "vector_docs_retrieved": len(vector_docs) if vector_docs else 0,
            
#             # Answer Quality Indicators (heuristic-based)
#             "contains_numbers": any(char.isdigit() for char in answer),
#             "contains_specific_routes": any(code in answer for code in ["CAI", "HBE", "JFK", "LAX"]),
#             "mentions_generation": any(gen in answer.lower() for gen in 
#                                       ["millennial", "gen x", "gen z", "baby boomer", "generation"]),
#             "answer_confidence": self.assess_confidence(answer),
#             "answer_completeness": self.assess_completeness(answer, question),
#         }
        
#         # Relevance scoring
#         if question in self.ground_truth:
#             gt = self.ground_truth[question]
#             metrics["expected_query_type"] = gt["query_type"]
#             metrics["entity_coverage"] = self.calculate_entity_coverage(answer, gt["expected_entities"])
        
#         self.results.append(metrics)
#         return metrics
    
#     def assess_confidence(self, answer):
#         """Assess confidence level based on language used"""
#         low_confidence_phrases = [
#             "could not be found", "not found", "unable to", 
#             "no data", "insufficient", "unclear"
#         ]
#         high_confidence_phrases = [
#             "specifically", "exactly", "precisely", "according to",
#             "the data shows", "results indicate"
#         ]
        
#         answer_lower = answer.lower()
        
#         if any(phrase in answer_lower for phrase in low_confidence_phrases):
#             return "low"
#         elif any(phrase in answer_lower for phrase in high_confidence_phrases):
#             return "high"
#         else:
#             return "medium"
    
#     def assess_completeness(self, answer, question):
#         """Assess if answer addresses all parts of the question"""
#         question_keywords = set(question.lower().split())
#         answer_words = set(answer.lower().split())
        
#         # Remove common stop words
#         stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "for", "to", "from"}
#         question_keywords -= stop_words
        
#         overlap = len(question_keywords & answer_words) / len(question_keywords) if question_keywords else 0
        
#         if overlap > 0.6:
#             return "high"
#         elif overlap > 0.3:
#             return "medium"
#         else:
#             return "low"
    
#     def calculate_entity_coverage(self, answer, expected_entities):
#         """Calculate what percentage of expected entities appear in answer"""
#         answer_lower = answer.lower()
#         found = sum(1 for entity in expected_entities if entity.lower() in answer_lower)
#         return found / len(expected_entities) if expected_entities else 0
    
#     def generate_comparison_report(self):
#         """Generate comprehensive comparison report"""
#         if not self.results:
#             return None
        
#         df = pd.DataFrame(self.results)
        
#         # Aggregate by model
#         model_stats = df.groupby('model').agg({
#             'response_time_seconds': ['mean', 'std', 'min', 'max'],
#             'answer_length_words': ['mean', 'std'],
#             'vector_docs_retrieved': 'mean',
#             'contains_numbers': 'sum',
#             'contains_specific_routes': 'sum',
#             'entity_coverage': 'mean'
#         }).round(3)
        
#         return {
#             'detailed_results': df,
#             'model_statistics': model_stats,
#             'raw_results': self.results
#         }

# def create_comparison_visualizations(comparison_data):
# # ... (create_comparison_visualizations methods remain the same) ...
#     """Create interactive visualizations for model comparison"""
    
#     df = comparison_data['detailed_results']
    
#     # 1. Response Time Comparison
#     fig_time = go.Figure()
#     for model in df['model'].unique():
#         model_data = df[df['model'] == model]
#         fig_time.add_trace(go.Box(
#             y=model_data['response_time_seconds'],
#             name=model,
#             boxmean='sd'
#         ))
#     fig_time.update_layout(
#         title="Response Time Comparison Across Models",
#         yaxis_title="Response Time (seconds)",
#         xaxis_title="Model"
#     )
    
#     # 2. Answer Quality Metrics
#     quality_metrics = df.groupby('model').agg({
#         'answer_completeness': lambda x: (x == 'high').sum() / len(x),
#         'answer_confidence': lambda x: (x == 'high').sum() / len(x),
#         'entity_coverage': 'mean'
#     }).reset_index()
    
#     fig_quality = go.Figure()
#     fig_quality.add_trace(go.Bar(
#         name='Completeness (High %)',
#         x=quality_metrics['model'],
#         y=quality_metrics['answer_completeness'] * 100
#     ))
#     fig_quality.add_trace(go.Bar(
#         name='Confidence (High %)',
#         x=quality_metrics['model'],
#         y=quality_metrics['answer_confidence'] * 100
#     ))
#     fig_quality.add_trace(go.Bar(
#         name='Entity Coverage (%)',
#         x=quality_metrics['model'],
#         y=quality_metrics['entity_coverage'] * 100
#     ))
#     fig_quality.update_layout(
#         title="Answer Quality Comparison",
#         yaxis_title="Percentage (%)",
#         xaxis_title="Model",
#         barmode='group'
#     )
    
#     # 3. Context Usage Pattern
#     context_usage = df.groupby('model').agg({
#         'used_cypher': 'sum',
#         'used_vector': 'sum',
#         'vector_docs_retrieved': 'mean'
#     }).reset_index()
    
#     fig_context = go.Figure()
#     fig_context.add_trace(go.Bar(
#         name='Used Cypher Query',
#         x=context_usage['model'],
#         y=context_usage['used_cypher']
#     ))
#     fig_context.add_trace(go.Bar(
#         name='Used Vector Store',
#         x=context_usage['model'],
#         y=context_usage['used_vector']
#     ))
#     fig_context.update_layout(
#         title="Context Source Usage by Model",
#         yaxis_title="Number of Times Used",
#         xaxis_title="Model",
#         barmode='group'
#     )
    
#     return {
#         'response_time': fig_time,
#         'quality_metrics': fig_quality,
#         'context_usage': fig_context
#     }


# def run_comparison_suite(config, vectorstore, test_questions, models_to_compare, context_mode):
#     """Run automated comparison across multiple models, with error handling for API keys."""
    
#     comparison = ModelComparison()
#     results_summary = []
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     total_tests = len(test_questions) * len(models_to_compare)
#     current_test = 0
    
#     for model_name in models_to_compare:
#         st.subheader(f"Testing: {model_name}")
#         model_data = MODEL_OPTIONS[model_name]
        
#         llm, retriever, qa_prompt = None, None, None
        
#         try:
#             # Initialize chain for this model
#             with st.spinner(f"Initializing {model_name}..."):
#                 llm, retriever, qa_prompt = initialize_rag_chain(
#                     config, vectorstore, model_data, context_mode
#                 )
#         except (ResourceExhausted, InvalidArgument) as e:
#             # Catch specific errors indicative of key/quota issues
#             st.error(f"**API Error for {model_name}:** Resource Exhausted or Invalid Key.")
#             st.warning(f"Please enter or update your API key in the sidebar and rerun the comparison.")
#             st.exception(e) # Show the full traceback for debugging if needed
            
#             # Skip the rest of the comparison since the model cannot be used
#             # Update the progress bar to show skipped models
#             current_test += (len(test_questions) * (len(models_to_compare) - models_to_compare.index(model_name))) / len(models_to_compare)
#             progress_bar.progress(current_test / total_tests)
#             continue
#         except Exception as e:
#             # Catch other unexpected initialization errors
#             st.error(f"Failed to initialize {model_name} due to an unexpected error: {str(e)}")
#             continue

        
#         if llm is None:
#             # This check is still useful if initialize_rag_chain returns None on failure
#             st.error(f"Failed to initialize {model_name}")
#             continue
        
#         for question in test_questions:
#             current_test += 1
#             progress_bar.progress(current_test / total_tests)
#             status_text.text(f"Testing {model_name} on question {current_test}/{total_tests}")
            
#             # Get Cypher context
#             start_time = time.time()
#             format_result, intent = extract_intent_and_entities(question)
            
#             # Get answer based on context mode
#             try:
#                 if context_mode == "cypher_only":
#                     chain = qa_prompt | llm
#                     result = chain.invoke({
#                         "input": question,
#                         "cypher_context": format_result
#                     })
#                     answer = result.content if hasattr(result, 'content') else str(result)
#                     retrieved_docs = []
                    
#                 elif context_mode == "vector_only":
#                     retrieved_docs = retriever.invoke(question)
#                     vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
#                     chain = qa_prompt | llm
#                     result = chain.invoke({
#                         "input": question,
#                         "vector_context": vector_structured
#                     })
#                     answer = result.content if hasattr(result, 'content') else str(result)
                    
#                 else:  # both
#                     retrieved_docs = retriever.invoke(question)
#                     vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
#                     chain = qa_prompt | llm
#                     result = chain.invoke({
#                         "input": question,
#                         "cypher_context": format_result,
#                         "vector_context": vector_structured
#                     })
#                     answer = result.content if hasattr(result, 'content') else str(result)
                
#                 response_time = time.time() - start_time
                
#                 # Evaluate response
#                 metrics = comparison.evaluate_response(
#                     question=question,
#                     answer=answer,
#                     model_name=model_name,
#                     context_mode=context_mode,
#                     response_time=response_time,
#                     cypher_result=format_result,
#                     vector_docs=retrieved_docs
#                 )
                
#                 results_summary.append({
#                     'model': model_name,
#                     'question': question[:50] + "...",
#                     'time': f"{response_time:.2f}s",
#                     'confidence': metrics['answer_confidence'],
#                     'completeness': metrics['answer_completeness']
#                 })
                
#             except Exception as e:
#                 # Catch API errors during invocation (e.g., query execution)
#                 st.error(f"Error testing {model_name} on question: {str(e)}")
#                 continue
    
#     progress_bar.progress(1.0)
#     status_text.text("Comparison complete!")
    
#     return comparison, results_summary

# def main():
#     st.set_page_config(page_title="🔬 Multi-Model RAG Comparison", layout="wide")
#     st.title("🔬 Multi-Model RAG Comparison System")
#     st.markdown("Compare performance of different LLMs on airline query tasks")
    
#     # 1. API Key Input & Update Logic
#     # Initialize session state for API Key
#     if 'api_key' not in st.session_state:
#         st.session_state['api_key'] = ''

#     # Sidebar for configuration
#     st.sidebar.header("Comparison Configuration")
    
#     st.sidebar.subheader("API Key Management")
#     new_api_key = st.sidebar.text_input(
#         "Enter your LLM API Key:", 
#         type="password", 
#         value=st.session_state['api_key']
#     )
    
#     if new_api_key and new_api_key != st.session_state['api_key']:
#         st.session_state['api_key'] = new_api_key
#         st.sidebar.success("API Key updated. Rerun comparison.")
        
#     # Check if key is available before proceeding
#     if not st.session_state['api_key']:
#         st.sidebar.warning("API Key is missing. Please enter your key to initialize models.")
#         # Only initialize components that don't depend on the key immediately
#         # setup_static_rag_components is called next, and it might need the key.
#         # We will pass the key to it via a conceptual 'config' update.
        
    
#     # Initialize components
#     # ASSUMPTION: The 'config' object is a dictionary or an object 
#     # that 'setup_static_rag_components' and 'initialize_rag_chain' uses 
#     # to access API keys/environment variables.
    
#     # Temporarily set the API key in the environment or a config dict 
#     # before calling the setup functions.
#     import os
#     os.environ['LLM_API_KEY'] = st.session_state['api_key']
    
#     # In a real app, you would need to update the specific key in your configuration:
#     # Example:
#     # config = {'llm_key': st.session_state['api_key']}
#     # config, vectorstore = setup_static_rag_components(config) 
    
#     config, vectorstore = setup_static_rag_components() # Assuming it picks up os.environ or is designed to handle this
    
    
#     if config is None or vectorstore is None:
#         st.error("Failed to initialize RAG components (Neo4j/Vectorstore). Check connection details.")
#         st.stop()
    
#     # Model selection (rest of the sidebar config remains the same)
#     st.sidebar.subheader("Select Models to Compare")
#     models_to_compare = []
#     for model_name in MODEL_OPTIONS.keys():
#         if st.sidebar.checkbox(model_name, value=True):
#             models_to_compare.append(model_name)
    
#     if len(models_to_compare) < 2:
#         st.warning("Please select at least 2 models to compare")
#         st.stop()
    
#     # Context mode selection
#     context_mode_name = st.sidebar.selectbox(
#         "Context Source:",
#         options=list(CONTEXT_OPTIONS.keys()),
#         index=2
#     )
#     context_mode = CONTEXT_OPTIONS[context_mode_name]
    
#     # Test questions
#     st.sidebar.subheader("Test Questions")
#     use_default = st.sidebar.checkbox("Use default test suite", value=True)
    
#     if use_default:
#         test_questions = [
#             "Which generation of passengers reported the lowest food satisfaction?",
#             "What is the average food satisfaction score for flights from LAX to IAX?",
#             "Show me the top 3 routes by passenger satisfaction",
#             "How many passengers from the Baby Boomer generation flew on route JFK to LAX?",
#             "What routes have the highest average delay?"
#         ]
#     else:
#         custom_questions = st.sidebar.text_area(
#             "Enter questions (one per line):",
#             height=200
#         )
#         test_questions = [q.strip() for q in custom_questions.split('\n') if q.strip()]
    
#     st.sidebar.markdown(f"**Total Tests:** {len(test_questions) * len(models_to_compare)}")
    
#     # Main content area
#     tab1, tab2, tab3 = st.tabs(["📊 Run Comparison", "📈 Visualizations", "📋 Detailed Results"])
    
#     with tab1:
#         st.header("Run Automated Comparison")
        
#         st.info(f"**Models Selected:** {', '.join(models_to_compare)}")
#         st.info(f"**Context Mode:** {context_mode_name}")
#         st.info(f"**Number of Test Questions:** {len(test_questions)}")
        
#         with st.expander("View Test Questions"):
#             for i, q in enumerate(test_questions, 1):
#                 st.write(f"{i}. {q}")
        
#         if st.button("🚀 Start Comparison", type="primary"):
#             # Ensure API Key is present before starting a comparison
#             if not st.session_state['api_key']:
#                 st.error("Cannot start comparison. Please enter your LLM API Key in the sidebar.")
#                 st.stop()
                
#             comparison, results_summary = run_comparison_suite(
#                 config, vectorstore, test_questions, models_to_compare, context_mode
#             )
            
#             # Store in session state
#             st.session_state['comparison'] = comparison
#             st.session_state['results_summary'] = results_summary
            
#             st.success("✅ Comparison completed!")
            
#             # Quick summary table
#             st.subheader("Quick Results Summary")
#             summary_df = pd.DataFrame(results_summary)
#             st.dataframe(summary_df, use_container_width=True)
    
#     with tab2:
#         st.header("Comparison Visualizations")
        
#         if 'comparison' in st.session_state:
#             comparison_data = st.session_state['comparison'].generate_comparison_report()
            
#             if comparison_data:
#                 visualizations = create_comparison_visualizations(comparison_data)
                
#                 st.plotly_chart(visualizations['response_time'], use_container_width=True)
#                 st.plotly_chart(visualizations['quality_metrics'], use_container_width=True)
#                 st.plotly_chart(visualizations['context_usage'], use_container_width=True)
                
#                 # Statistical summary
#                 st.subheader("Statistical Summary")
#                 st.dataframe(comparison_data['model_statistics'], use_container_width=True)
#         else:
#             st.info("Run a comparison first to see visualizations")
    
#     with tab3:
#         st.header("Detailed Results")
        
#         if 'comparison' in st.session_state:
#             comparison_data = st.session_state['comparison'].generate_comparison_report()
            
#             if comparison_data:
#                 df = comparison_data['detailed_results']
                
#                 # Filters
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     selected_model = st.selectbox(
#                         "Filter by Model:",
#                         options=["All"] + list(df['model'].unique())
#                     )
#                 with col2:
#                     selected_confidence = st.selectbox(
#                         "Filter by Confidence:",
#                         options=["All", "high", "medium", "low"]
#                     )
                
#                 # Apply filters
#                 filtered_df = df.copy()
#                 if selected_model != "All":
#                     filtered_df = filtered_df[filtered_df['model'] == selected_model]
#                 if selected_confidence != "All":
#                     filtered_df = filtered_df[filtered_df['answer_confidence'] == selected_confidence]
                
#                 # Display detailed results
#                 st.dataframe(filtered_df, use_container_width=True)
                
#                 # Export option
#                 csv = filtered_df.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Results as CSV",
#                     data=csv,
#                     file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                     mime="text/csv"
#                 )
                
#                 # Individual answer inspection
#                 st.subheader("Inspect Individual Answers")
#                 for idx, row in filtered_df.iterrows():
#                     with st.expander(f"{row['model']} - {row['question'][:60]}..."):
#                         st.markdown(f"**Question:** {row['question']}")
#                         st.markdown(f"**Answer:** {row['answer']}")
                        
#                         col1, col2, col3, col4 = st.columns(4)
#                         col1.metric("Response Time", f"{row['response_time_seconds']:.2f}s")
#                         col2.metric("Confidence", row['answer_confidence'])
#                         col3.metric("Completeness", row['answer_completeness'])
#                         col4.metric("Word Count", row['answer_length_words'])
#         else:
#             st.info("Run a comparison first to see detailed results")

# if __name__ == "__main__":
#     main()



"""
UPDATED run_comparison_suite function and evaluate_response integration
Add this to your existing comparison code
"""

import tiktoken
import streamlit as st
import time

def count_tokens(text, model_name="Gemini 2.5 Flash"):
    """Count tokens in text - adapted for your models"""
    try:
        # Use cl100k_base for all models (approximation)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback: approximate tokens as words * 1.3
        return int(len(text.split()) * 1.3)

def calculate_cost(input_tokens, output_tokens, model_name):
    """Calculate cost based on token usage"""
    model_pricing = {
        "Gemini 2.5 Flash": {"input": 0.075, "output": 0.30},
        "Mistral 7B Instruct": {"input": 0.0, "output": 0.0},  # Free
        "Gemma 2B IT": {"input": 0.0, "output": 0.0}  # Free
    }
    
    if model_name not in model_pricing:
        return 0.0
    
    pricing = model_pricing[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost

def calculate_accuracy(question, answer, ground_truth_item):
    """Calculate accuracy score based on answer type"""
    answer_type = ground_truth_item["answer_type"]
    expected_entities = ground_truth_item["expected_entities"]
    answer_lower = answer.lower()
    
    # Entity presence score (0-1)
    entity_score = sum(1 for entity in expected_entities 
                      if entity.lower() in answer_lower) / len(expected_entities)
    
    # Type-specific accuracy
    if answer_type == "categorical":
        expected_val = ground_truth_item["expected_value"]
        if expected_val and expected_val.lower() in answer_lower:
            type_score = 1.0
        else:
            type_score = 0.0
    
    elif answer_type == "numeric":
        type_score = 1.0 if any(char.isdigit() for char in answer) else 0.0
    
    elif answer_type == "list":
        list_indicators = ["1.", "2.", "3.", "•", "-", "top", "first", "second"]
        type_score = 1.0 if any(ind in answer_lower for ind in list_indicators) else 0.0
    
    else:
        type_score = 0.5
    
    # Combined accuracy score
    accuracy = (entity_score * 0.6) + (type_score * 0.4)
    return round(accuracy, 3)

def calculate_correctness(answer, cypher_result, vector_docs):
    """Calculate correctness based on context usage and answer quality"""
    correctness_score = 0.0
    
    # 1. Used appropriate context (0.3)
    if cypher_result and "No results found" not in cypher_result:
        correctness_score += 0.15
    if vector_docs and len(vector_docs) > 0:
        correctness_score += 0.15
    
    # 2. Answer is not evasive (0.3)
    evasive_phrases = [
        "could not be found", "not found", "unable to", 
        "no data", "insufficient", "i don't have", "cannot find"
    ]
    if not any(phrase in answer.lower() for phrase in evasive_phrases):
        correctness_score += 0.3
    
    # 3. Answer has substantive content (0.2)
    if len(answer.split()) > 20:
        correctness_score += 0.2
    
    # 4. Answer contains specific information (0.2)
    specific_indicators = [
        any(char.isdigit() for char in answer),
        any(code in answer for code in ["CAI", "HBE", "JFK", "LAX", "IAX"]),
        "%" in answer or "average" in answer.lower()
    ]
    correctness_score += 0.2 * (sum(specific_indicators) / len(specific_indicators))
    
    return round(correctness_score, 3)


def run_comparison_suite_with_metrics(config, vectorstore, test_questions, models_to_compare, context_mode):
    """
    UPDATED: Run automated comparison with ALL quantitative metrics
    """
    
    # Enhanced ground truth for your specific questions
    ground_truth = {
        "What are the different passenger generations?": {
            "expected_value": ["Millennial","Gen X","Gen Y","Boomer", "Silent","NBK"],
            "expected_entities": ["generation", "Millennial", "Gen X", "Gen Y", "Boomer", "Silent","NBK"],
            "query_type": "aggregation",
            "answer_type": "categorical"
        },
        "What is possible destinations from LAX?": {
            "expected_value": [
    "IAX2", "ORX3", "BWX4", "DEX5", "EWX6", "IAX7", "EWX8", 
    "IAX9", "EWX10", "EWX11", "IAX12", "IAX13", "EWX14", "IAX15", 
    "AUX16", "ORX17", "IAX18", "LIX19", "IAX20", "PHX21", "ORX22", 
    "JFX23", "DEX24", "HNX25", "OGX26", "ORX27", "KOX28", "SFX29", 
    "EWX30", "SFX31", "IAX32", "ORX33", "OGX34", "ORX35", "BOX36", 
    "ORX37", "OGX38", "IAX39", "DEX40", "SFX41", "LAX42", "EWX43", 
    "EWX44", "IAX45", "IAX46", "SFX47", "IAX48", "EWX49", "EWX50", 
    "SYX51", "IAX52", "TPX53", "HNX54", "SFX55", "EWX56", "IAX57", 
    "LHX58", "HNX59", "MCX60", "ORX61", "IAX62", "EWX63", "EWX64", 
    "SFX65", "IAX66", "SFX67", "BWX68", "EWX69", "EWX70", "ORX71", 
    "EWX72", "ORX73", "IAX74", "NRX75", "ORX76", "CUX77", "DEX78", 
    "EWX79", "IAX80", "IAX81", "YVX82", "IAX83", "HNX" # Assuming HNX is missing its number
],
            "expected_entities": ["destination","LAX"],
            "query_type": "route_specific",
            "answer_type": "numeric"
        },
        "What are the different loyalty levels?": {
            "expected_value": ["NBK", "premier platinum", "global services", "premier 1k", "premier silver", "premier gold", "non-elite"],
            "expected_entities": ["loyalty level", "NBK", "premier platinum", "global services", "premier 1k", "premier silver", "premier gold", "non-elite"],
            "query_type": "categorical",
            "answer_type": "list"
        },
        "What is the number of flights from LAX to IAX": {
            "expected_value": 21,
            "expected_entities": ["flights", "LAX", "IAX"],
            "query_type": "count",
            "answer_type": "numeric"
        },
        "What are the top 10 routes that have the highest average delay?": {
            "expected_value": [("DEX", "JAX", 284.02),
    ("SFX", "SIX", 220.33333333333331),
    ("TLX", "IAX", 147.33333333333334),
    ("POX", "IAX", 134.05),
    ("DEX", "FRX", 131.06),
    ("RDX", "ORX", 107.07),
    ("GSX", "DEX", 92.58),
    ("OKX", "IAX", 83.59),
    ("OKX", "SAX", 83.51),
    ("IAX", "RDX", 83.0)],
            "expected_entities": ["routes", "delay", "highest", "average"],
            "query_type": "ranking",
            "answer_type": "list"
        }
    }
    
    results = []
    results_summary = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tests = len(test_questions) * len(models_to_compare)
    current_test = 0
    
    for model_name in models_to_compare:
        st.subheader(f"Testing: {model_name}")
        model_data = MODEL_OPTIONS[model_name]
        
        try:
            with st.spinner(f"Initializing {model_name}..."):
                llm, retriever, qa_prompt = initialize_rag_chain(
                    config, vectorstore, model_data, context_mode
                )
        except Exception as e:
            st.error(f"Failed to initialize {model_name}: {str(e)}")
            continue
        
        if llm is None:
            st.error(f"Failed to initialize {model_name}")
            continue
        
        for question in test_questions:
            current_test += 1
            progress_bar.progress(current_test / total_tests)
            status_text.text(f"Testing {model_name} on question {current_test}/{total_tests}")
            
            # Start timing
            start_time = time.time()
            
            # Get Cypher context
            format_result, intent , _ , _ = extract_intent_and_entities(question)
            
            # Prepare input text for token counting
            input_text = question
            cypher_context_text = format_result if format_result else ""
            
            try:
                if context_mode == "cypher_only":
                    # Combine all input for token counting
                    full_input = f"{input_text}\n{cypher_context_text}"
                    input_tokens = count_tokens(full_input, model_name)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": question,
                        "cypher_context": format_result
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    retrieved_docs = []
                    
                elif context_mode == "vector_only":
                    retrieved_docs = retriever.invoke(question)
                    vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
                    # Combine all input for token counting
                    full_input = f"{input_text}\n{vector_structured}"
                    input_tokens = count_tokens(full_input, model_name)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": question,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    
                else:  # both
                    retrieved_docs = retriever.invoke(question)
                    vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
                    # Combine all input for token counting
                    full_input = f"{input_text}\n{cypher_context_text}\n{vector_structured}"
                    input_tokens = count_tokens(full_input, model_name)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": question,
                        "cypher_context": format_result,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Count output tokens
                output_tokens = count_tokens(answer, model_name)
                total_tokens = input_tokens + output_tokens
                
                # Calculate cost
                cost = calculate_cost(input_tokens, output_tokens, model_name)
                
                # Calculate accuracy and correctness
                accuracy = 0.0
                correctness = calculate_correctness(answer, format_result, retrieved_docs)
                
                if question in ground_truth:
                    gt = ground_truth[question]
                    accuracy = calculate_accuracy(question, answer, gt)
                
                # Calculate entity coverage (from your original code)
                if question in ground_truth:
                    expected_entities = ground_truth[question]["expected_entities"]
                    answer_lower = answer.lower()
                    found = sum(1 for entity in expected_entities if entity.lower() in answer_lower)
                    entity_coverage = round(found / len(expected_entities), 3) if expected_entities else 0
                else:
                    entity_coverage = 0
                
                # Store ALL metrics
                metrics = {
                    # Identification
                    "model": model_name,
                    "context_mode": context_mode,
                    "question": question,
                    "answer": answer,
                    
                    # ===== QUANTITATIVE METRICS =====
                    
                    # 1. Performance Metrics
                    "response_time_seconds": round(response_time, 3),
                    "tokens_per_second": round(output_tokens / response_time, 2) if response_time > 0 else 0,
                    
                    # 2. Token Usage
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    
                    # 3. Cost
                    "cost_usd": round(cost, 6),
                    "cost_per_1k_output_tokens": round((cost / output_tokens * 1000), 6) if output_tokens > 0 else 0,
                    
                    # 4. Accuracy & Correctness
                    "accuracy_score": accuracy,
                    "correctness_score": correctness,
                    "combined_quality_score": round((accuracy + correctness) / 2, 3),
                    "entity_coverage": entity_coverage,
                    
                    # 5. Answer Characteristics
                    "answer_length_words": len(answer.split()),
                    "answer_length_tokens": output_tokens,
                    
                    # 6. Context Usage
                    "used_cypher": bool(format_result and "No results found" not in format_result),
                    "used_vector": bool(retrieved_docs and len(retrieved_docs) > 0),
                    "vector_docs_retrieved": len(retrieved_docs) if retrieved_docs else 0,
                }
                
                results.append(metrics)
                
                results_summary.append({
                    'model': model_name,
                    'question': question[:50] + "...",
                    'time_s': f"{response_time:.2f}",
                    'tokens': total_tokens,
                    'cost_usd': f"${cost:.6f}",
                    'accuracy': f"{accuracy:.2%}",
                    'correctness': f"{correctness:.2%}",
                    'quality': f"{metrics['combined_quality_score']:.2%}"
                })
                
            except Exception as e:
                st.error(f"Error testing {model_name} on question: {str(e)}")
                continue
    
    progress_bar.progress(1.0)
    status_text.text("Comparison complete!")
    
    # Return both the results list and summary
    return results, results_summary


# ===== SUMMARY DISPLAY FUNCTION =====
def display_quantitative_summary_dashboard(results_df):
    """
    Display comprehensive quantitative metrics dashboard
    """
    st.subheader("📊 Quantitative Metrics Summary")
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        with st.expander(f"**{model}** - Complete Metrics", expanded=True):
            # Create 4 columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**⚡ Performance**")
                st.metric("Avg Response Time", f"{model_data['response_time_seconds'].mean():.2f}s")
                st.metric("Tokens/Second", f"{model_data['tokens_per_second'].mean():.1f}")
            
            with col2:
                st.markdown("**🔢 Token Usage**")
                st.metric("Avg Total Tokens", f"{model_data['total_tokens'].mean():.0f}")
                st.metric("Avg Output Tokens", f"{model_data['output_tokens'].mean():.0f}")
            
            with col3:
                st.markdown("**💰 Cost**")
                total_cost = model_data['cost_usd'].sum()
                avg_cost = model_data['cost_usd'].mean()
                
                if total_cost == 0:
                    st.metric("Total Cost", "FREE")
                    st.metric("Avg Cost/Query", "FREE")
                else:
                    st.metric("Total Cost", f"${total_cost:.6f}")
                    st.metric("Avg Cost/Query", f"${avg_cost:.6f}")
            
            with col4:
                st.markdown("**✅ Quality Scores**")
                st.metric("Accuracy", f"{model_data['accuracy_score'].mean():.1%}")
                st.metric("Correctness", f"{model_data['correctness_score'].mean():.1%}")


# ===== EXAMPLE USAGE IN MAIN COMPARISON TAB =====
def example_tab1_integration():
    """
    Example of how to integrate this in your Tab 1
    """
    with tab1:
        st.header("Run Automated Comparison")
        
        if st.button("🚀 Start Comparison", type="primary"):
            if not st.session_state.get('api_key'):
                st.error("Cannot start comparison. Please enter your LLM API Key in the sidebar.")
                st.stop()
            
            # Use the updated function
            results_list, results_summary = run_comparison_suite_with_metrics(
                config, vectorstore, test_questions, models_to_compare, context_mode
            )
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results_list)
            
            # Store in session state
            st.session_state['comparison_results'] = results_df
            st.session_state['results_summary'] = results_summary
            
            st.success("✅ Comparison completed!")
            
            # Display summary table
            st.subheader("Quick Results Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Display quantitative dashboard
            display_quantitative_summary_dashboard(results_df)


"""
STANDALONE MAIN FUNCTION
Add this to the bottom of your existing comparison code file
Do NOT modify any of your existing code - just add this at the end
"""

# def main():
#     st.set_page_config(page_title="🔬 Multi-Model RAG Comparison", layout="wide")
#     st.title("🔬 Multi-Model RAG Comparison System")
#     st.markdown("Compare performance of different LLMs on airline query tasks")
    
#     # Sidebar Configuration
#     st.sidebar.header("Comparison Configuration")
    
#     # Initialize components from your existing code
#     config, vectorstore = setup_static_rag_components()
    
#     if config is None or vectorstore is None:
#         st.error("Failed to initialize RAG components. Check Neo4j connection.")
#         st.stop()
    
#     # Model selection
#     st.sidebar.subheader("Select Models to Compare")
#     models_to_compare = []
#     for model_name in MODEL_OPTIONS.keys():
#         if st.sidebar.checkbox(model_name, value=True):
#             models_to_compare.append(model_name)
    
#     if len(models_to_compare) < 1:
#         st.warning("Please select at least 1 model to compare")
#         st.stop()
    
#     # Context mode selection
#     context_mode_name = st.sidebar.selectbox(
#         "Context Source:",
#         options=list(CONTEXT_OPTIONS.keys()),
#         index=2
#     )
#     context_mode = CONTEXT_OPTIONS[context_mode_name]
    
#     # Test questions
#     st.sidebar.subheader("Test Questions")
#     use_default = st.sidebar.checkbox("Use default test suite", value=True)
    
#     if use_default:
#         test_questions = [
#             "What are the different passenger generations?",
#             "What is possible destinations from LAX?",
#             "What are the different loyalty levels?",
#             "What is the number of flights from LAX to IAX",
#             "What are the top 10 routes that have the highest average delay?"
#         ]
#     else:
#         custom_questions = st.sidebar.text_area(
#             "Enter questions (one per line):",
#             height=200
#         )
#         test_questions = [q.strip() for q in custom_questions.split('\n') if q.strip()]
    
#     st.sidebar.markdown(f"**Total Tests:** {len(test_questions) * len(models_to_compare)}")
    
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs(["📊 Run Comparison", "📈 Visualizations", "📋 Detailed Results"])
    
#     # ===== TAB 1: RUN COMPARISON =====
#     with tab1:
#         st.header("Run Automated Comparison")
        
#         st.info(f"**Models Selected:** {', '.join(models_to_compare)}")
#         st.info(f"**Context Mode:** {context_mode_name}")
#         st.info(f"**Number of Test Questions:** {len(test_questions)}")
        
#         with st.expander("View Test Questions"):
#             for i, q in enumerate(test_questions, 1):
#                 st.write(f"{i}. {q}")
        
#         if st.button("🚀 Start Comparison", type="primary"):
#             # Call the comparison function from your code
#             results_list, results_summary = run_comparison_suite_with_metrics(
#                 config, vectorstore, test_questions, models_to_compare, context_mode
#             )
            
#             # Convert to DataFrame
#             results_df = pd.DataFrame(results_list)
            
#             # Store in session state
#             st.session_state['comparison_results'] = results_df
#             st.session_state['results_summary'] = results_summary
            
#             st.success("✅ Comparison completed!")
            
#             # Display summary table
#             st.subheader("Quick Results Summary")
#             summary_df = pd.DataFrame(results_summary)
#             st.dataframe(summary_df, use_container_width=True)
            
#             # Display quantitative dashboard
#             display_quantitative_summary_dashboard(results_df)
    
#     # ===== TAB 2: VISUALIZATIONS =====
#     with tab2:
#         st.header("Comparison Visualizations")
        
#         if 'comparison_results' in st.session_state:
#             results_df = st.session_state['comparison_results']
            
#             # 1. Response Time vs Cost Trade-off
#             st.subheader("Response Time vs Cost Trade-off")
#             fig_tradeoff = go.Figure()
#             for model in results_df['model'].unique():
#                 model_data = results_df[results_df['model'] == model]
#                 fig_tradeoff.add_trace(go.Scatter(
#                     x=model_data['response_time_seconds'],
#                     y=model_data['cost_usd'],
#                     mode='markers',
#                     name=model,
#                     marker=dict(size=model_data['combined_quality_score']*20),
#                     text=[f"Quality: {q:.2f}" for q in model_data['combined_quality_score']],
#                     hovertemplate="<b>%{fullData.name}</b><br>" +
#                                  "Time: %{x:.2f}s<br>" +
#                                  "Cost: $%{y:.6f}<br>" +
#                                  "%{text}<extra></extra>"
#                 ))
#             fig_tradeoff.update_layout(
#                 title="Response Time vs Cost (bubble size = quality score)",
#                 xaxis_title="Response Time (seconds)",
#                 yaxis_title="Cost (USD)",
#                 height=500
#             )
#             st.plotly_chart(fig_tradeoff, use_container_width=True)
            
#             # 2. Quality Metrics Bar Chart
#             st.subheader("Quality Metrics Comparison")
#             quality_agg = results_df.groupby('model').agg({
#                 'accuracy_score': 'mean',
#                 'correctness_score': 'mean',
#                 'combined_quality_score': 'mean'
#             }).reset_index()
            
#             fig_quality = go.Figure()
#             fig_quality.add_trace(go.Bar(
#                 name='Accuracy',
#                 x=quality_agg['model'],
#                 y=quality_agg['accuracy_score']*100
#             ))
#             fig_quality.add_trace(go.Bar(
#                 name='Correctness',
#                 x=quality_agg['model'],
#                 y=quality_agg['correctness_score']*100
#             ))
#             fig_quality.add_trace(go.Bar(
#                 name='Combined Quality',
#                 x=quality_agg['model'],
#                 y=quality_agg['combined_quality_score']*100
#             ))
#             fig_quality.update_layout(
#                 title="Quality Metrics Comparison (%)",
#                 yaxis_title="Score (%)",
#                 xaxis_title="Model",
#                 barmode='group',
#                 height=500
#             )
#             st.plotly_chart(fig_quality, use_container_width=True)
            
#             # 3. Token Usage
#             st.subheader("Token Usage Comparison")
#             token_agg = results_df.groupby('model').agg({
#                 'input_tokens': 'mean',
#                 'output_tokens': 'mean'
#             }).reset_index()
            
#             fig_tokens = go.Figure()
#             fig_tokens.add_trace(go.Bar(
#                 name='Input Tokens',
#                 x=token_agg['model'],
#                 y=token_agg['input_tokens']
#             ))
#             fig_tokens.add_trace(go.Bar(
#                 name='Output Tokens',
#                 x=token_agg['model'],
#                 y=token_agg['output_tokens']
#             ))
#             fig_tokens.update_layout(
#                 title="Average Token Usage by Model",
#                 yaxis_title="Number of Tokens",
#                 xaxis_title="Model",
#                 barmode='stack',
#                 height=500
#             )
#             st.plotly_chart(fig_tokens, use_container_width=True)
            
#             # 4. Response Time Distribution
#             st.subheader("Response Time Distribution")
#             fig_time = go.Figure()
#             for model in results_df['model'].unique():
#                 model_data = results_df[results_df['model'] == model]
#                 fig_time.add_trace(go.Box(
#                     y=model_data['response_time_seconds'],
#                     name=model,
#                     boxmean='sd'
#                 ))
#             fig_time.update_layout(
#                 title="Response Time Distribution by Model",
#                 yaxis_title="Response Time (seconds)",
#                 xaxis_title="Model",
#                 height=500
#             )
#             st.plotly_chart(fig_time, use_container_width=True)
            
#             # Statistical Summary Table
#             st.subheader("Statistical Summary")
#             summary_stats = results_df.groupby('model').agg({
#                 'response_time_seconds': ['mean', 'std', 'min', 'max'],
#                 'total_tokens': ['mean', 'std'],
#                 'cost_usd': ['mean', 'sum'],
#                 'accuracy_score': 'mean',
#                 'correctness_score': 'mean',
#                 'combined_quality_score': 'mean'
#             }).round(3)
#             st.dataframe(summary_stats, use_container_width=True)
            
#         else:
#             st.info("Run a comparison first to see visualizations")
    
#     # ===== TAB 3: DETAILED RESULTS =====
#     with tab3:
#         st.header("Detailed Results")
        
#         if 'comparison_results' in st.session_state:
#             results_df = st.session_state['comparison_results']
            
#             # Filters
#             col1, col2 = st.columns(2)
#             with col1:
#                 selected_model = st.selectbox(
#                     "Filter by Model:",
#                     options=["All"] + list(results_df['model'].unique())
#                 )
#             with col2:
#                 selected_question = st.selectbox(
#                     "Filter by Question:",
#                     options=["All"] + list(results_df['question'].unique())
#                 )
            
#             # Apply filters
#             filtered_df = results_df.copy()
#             if selected_model != "All":
#                 filtered_df = filtered_df[filtered_df['model'] == selected_model]
#             if selected_question != "All":
#                 filtered_df = filtered_df[filtered_df['question'] == selected_question]
            
#             # Display filtered results
#             st.dataframe(filtered_df, use_container_width=True)
            
#             # Export option
#             csv = filtered_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Results as CSV",
#                 data=csv,
#                 file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
            
#             # Individual answer inspection
#             st.subheader("Inspect Individual Answers")
#             for idx, row in filtered_df.iterrows():
#                 with st.expander(f"{row['model']} - {row['question'][:60]}..."):
#                     st.markdown(f"**Question:** {row['question']}")
#                     st.markdown(f"**Answer:** {row['answer']}")
                    
#                     col1, col2, col3, col4 = st.columns(4)
#                     col1.metric("Response Time", f"{row['response_time_seconds']:.2f}s")
#                     col2.metric("Total Tokens", f"{row['total_tokens']}")
#                     col3.metric("Cost", f"${row['cost_usd']:.6f}" if row['cost_usd'] > 0 else "FREE")
#                     col4.metric("Quality Score", f"{row['combined_quality_score']:.2%}")
                    
#                     # Additional metrics
#                     st.markdown("---")
#                     col5, col6, col7, col8 = st.columns(4)
#                     col5.metric("Accuracy", f"{row['accuracy_score']:.2%}")
#                     col6.metric("Correctness", f"{row['correctness_score']:.2%}")
#                     col7.metric("Entity Coverage", f"{row['entity_coverage']:.2%}")
#                     col8.metric("Tokens/Sec", f"{row['tokens_per_second']:.1f}")
#         else:
#             st.info("Run a comparison first to see detailed results")


# if __name__ == "__main__":
#     main()


def main():
    st.set_page_config(page_title="🔬 Multi-Model RAG Comparison", layout="wide")
    st.title("🔬 Multi-Model RAG Comparison System")
    st.markdown("Compare performance of different LLMs on airline query tasks")
    
    # Sidebar Configuration
    st.sidebar.header("Comparison Configuration")
    
    # Initialize components from your existing code
    config, vectorstore = setup_static_rag_components()
    
    if config is None or vectorstore is None:
        st.error("Failed to initialize RAG components. Check Neo4j connection.")
        st.stop()
    
    # Model selection
    st.sidebar.subheader("Select Models to Compare")
    models_to_compare = []
    for model_name in MODEL_OPTIONS.keys():
        if st.sidebar.checkbox(model_name, value=True):
            models_to_compare.append(model_name)
    
    if len(models_to_compare) < 1:
        st.warning("Please select at least 1 model to compare")
        st.stop()
    
    # Context mode selection
    context_mode_name = st.sidebar.selectbox(
        "Context Source:",
        options=list(CONTEXT_OPTIONS.keys()),
        index=2
    )
    context_mode = CONTEXT_OPTIONS[context_mode_name]
    
    # Test questions
    st.sidebar.subheader("Test Questions")
    use_default = st.sidebar.checkbox("Use default test suite", value=True)
    
    if use_default:
        test_questions = [
            "What are the different passenger generations?",
            "What is possible destinations from LAX?",
            "What are the different loyalty levels?",
            "What is the number of flights from LAX to IAX",
            "What are the top 10 routes that have the highest average delay?"
        ]
    else:
        custom_questions = st.sidebar.text_area(
            "Enter questions (one per line):",
            height=200
        )
        test_questions = [q.strip() for q in custom_questions.split('\n') if q.strip()]
    
    st.sidebar.markdown(f"**Total Tests:** {len(test_questions) * len(models_to_compare)}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Run Comparison", "📈 Visualizations", "📋 Detailed Results"])
    
    # ===== TAB 1: RUN COMPARISON =====
    with tab1:
        st.header("Run Automated Comparison")
        
        st.info(f"**Models Selected:** {', '.join(models_to_compare)}")
        st.info(f"**Context Mode:** {context_mode_name}")
        st.info(f"**Number of Test Questions:** {len(test_questions)}")
        
        with st.expander("View Test Questions"):
            for i, q in enumerate(test_questions, 1):
                st.write(f"{i}. {q}")
        
        if st.button("🚀 Start Comparison", type="primary"):
            # Call the comparison function from your code
            results_list, results_summary = run_comparison_suite_with_metrics(
                config, vectorstore, test_questions, models_to_compare, context_mode
            )
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results_list)
            
            # Store in session state
            st.session_state['comparison_results'] = results_df
            st.session_state['results_summary'] = results_summary
            
            st.success("✅ Comparison completed!")
            
            # Display summary table
            st.subheader("Quick Results Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Display quantitative dashboard
            display_quantitative_summary_dashboard(results_df)
    
    # ===== TAB 2: VISUALIZATIONS =====
    with tab2:
        st.header("Comparison Visualizations")
        
        if 'comparison_results' in st.session_state:
            results_df = st.session_state['comparison_results']
            
            # 1. Response Time vs Cost Trade-off
            st.subheader("Response Time vs Cost Trade-off")
            fig_tradeoff = go.Figure()
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                fig_tradeoff.add_trace(go.Scatter(
                    x=model_data['response_time_seconds'],
                    y=model_data['cost_usd'],
                    mode='markers',
                    name=model,
                    marker=dict(size=model_data['combined_quality_score']*20),
                    text=[f"Quality: {q:.2f}" for q in model_data['combined_quality_score']],
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                 "Time: %{x:.2f}s<br>" +
                                 "Cost: $%{y:.6f}<br>" +
                                 "%{text}<extra></extra>"
                ))
            fig_tradeoff.update_layout(
                title="Response Time vs Cost (bubble size = quality score)",
                xaxis_title="Response Time (seconds)",
                yaxis_title="Cost (USD)",
                height=500
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            # 2. Quality Metrics Bar Chart
            st.subheader("Quality Metrics Comparison")
            quality_agg = results_df.groupby('model').agg({
                'accuracy_score': 'mean',
                'correctness_score': 'mean',
                'combined_quality_score': 'mean'
            }).reset_index()
            
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Bar(
                name='Accuracy',
                x=quality_agg['model'],
                y=quality_agg['accuracy_score']*100
            ))
            fig_quality.add_trace(go.Bar(
                name='Correctness',
                x=quality_agg['model'],
                y=quality_agg['correctness_score']*100
            ))
            fig_quality.add_trace(go.Bar(
                name='Combined Quality',
                x=quality_agg['model'],
                y=quality_agg['combined_quality_score']*100
            ))
            fig_quality.update_layout(
                title="Quality Metrics Comparison (%)",
                yaxis_title="Score (%)",
                xaxis_title="Model",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig_quality, use_container_width=True)
            
            # 3. Token Usage
            st.subheader("Token Usage Comparison")
            token_agg = results_df.groupby('model').agg({
                'input_tokens': 'mean',
                'output_tokens': 'mean'
            }).reset_index()
            
            fig_tokens = go.Figure()
            fig_tokens.add_trace(go.Bar(
                name='Input Tokens',
                x=token_agg['model'],
                y=token_agg['input_tokens']
            ))
            fig_tokens.add_trace(go.Bar(
                name='Output Tokens',
                x=token_agg['model'],
                y=token_agg['output_tokens']
            ))
            fig_tokens.update_layout(
                title="Average Token Usage by Model",
                yaxis_title="Number of Tokens",
                xaxis_title="Model",
                barmode='stack',
                height=500
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
            
            # 4. Response Time Distribution
            st.subheader("Response Time Distribution")
            fig_time = go.Figure()
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                fig_time.add_trace(go.Box(
                    y=model_data['response_time_seconds'],
                    name=model,
                    boxmean='sd'
                ))
            fig_time.update_layout(
                title="Response Time Distribution by Model",
                yaxis_title="Response Time (seconds)",
                xaxis_title="Model",
                height=500
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Statistical Summary Table
            st.subheader("Statistical Summary")
            summary_stats = results_df.groupby('model').agg({
                'response_time_seconds': ['mean', 'std', 'min', 'max'],
                'total_tokens': ['mean', 'std'],
                'cost_usd': ['mean', 'sum'],
                'accuracy_score': 'mean',
                'correctness_score': 'mean',
                'combined_quality_score': 'mean'
            }).round(3)
            st.dataframe(summary_stats, use_container_width=True)
            
        else:
            st.info("Run a comparison first to see visualizations")
    
    # ===== TAB 3: DETAILED RESULTS =====
    with tab3:
        st.header("Detailed Results")
        
        if 'comparison_results' in st.session_state:
            results_df = st.session_state['comparison_results']
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox(
                    "Filter by Model:",
                    options=["All"] + list(results_df['model'].unique())
                )
            with col2:
                selected_question = st.selectbox(
                    "Filter by Question:",
                    options=["All"] + list(results_df['question'].unique())
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            if selected_model != "All":
                filtered_df = filtered_df[filtered_df['model'] == selected_model]
            if selected_question != "All":
                filtered_df = filtered_df[filtered_df['question'] == selected_question]
            
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Individual answer inspection
            st.subheader("Inspect Individual Answers")
            for idx, row in filtered_df.iterrows():
                with st.expander(f"{row['model']} - {row['question'][:60]}..."):
                    st.markdown(f"**Question:** {row['question']}")
                    st.markdown(f"**Answer:** {row['answer']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Response Time", f"{row['response_time_seconds']:.2f}s")
                    col2.metric("Total Tokens", f"{row['total_tokens']}")
                    col3.metric("Cost", f"${row['cost_usd']:.6f}" if row['cost_usd'] > 0 else "FREE")
                    col4.metric("Quality Score", f"{row['combined_quality_score']:.2%}")
                    
                    # Additional metrics
                    st.markdown("---")
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Accuracy", f"{row['accuracy_score']:.2%}")
                    col6.metric("Correctness", f"{row['correctness_score']:.2%}")
                    col7.metric("Entity Coverage", f"{row['entity_coverage']:.2%}")
                    col8.metric("Tokens/Sec", f"{row['tokens_per_second']:.1f}")
        else:
            st.info("Run a comparison first to see detailed results")


if __name__ == "__main__":
    main()