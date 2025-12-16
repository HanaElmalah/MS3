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

class ModelComparison:
    """Handles comparison metrics and evaluation for multiple LLM models"""
    
    def __init__(self):
        self.results = []
        self.ground_truth = self.load_ground_truth()
    
    def load_ground_truth(self):
        """Define ground truth answers for evaluation questions"""
        return {
            "Which generation of passengers reported the lowest food satisfaction?": {
                "expected_answer": "Millennials",
                "expected_entities": ["Millennials", "generation"],
                "query_type": "aggregation"
            },
            "What is the average food satisfaction score for flights from CAI to HBE?": {
                "expected_answer": "specific numeric value",
                "expected_entities": ["CAI", "HBE", "food satisfaction"],
                "query_type": "route_specific"
            },
            "Show me the top 3 routes by passenger satisfaction": {
                "expected_answer": "list of routes with satisfaction scores",
                "expected_entities": ["routes", "satisfaction", "top"],
                "query_type": "ranking"
            },
            "How many passengers from the Baby Boomer generation flew on route JFK to LAX?": {
                "expected_answer": "specific count",
                "expected_entities": ["Baby Boomer", "JFK", "LAX"],
                "query_type": "count"
            },
            "What routes have the highest average delay?": {
                "expected_answer": "routes with delay information",
                "expected_entities": ["routes", "delay", "highest"],
                "query_type": "ranking"
            }
        }
    
    def evaluate_response(self, question, answer, model_name, context_mode, 
                         response_time, cypher_result, vector_docs):
        """Comprehensive evaluation of a model's response"""
        
        metrics = {
            "model": model_name,
            "context_mode": context_mode,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            
            # Quantitative Metrics
            "response_time_seconds": response_time,
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            
            # Context Usage Metrics
            "used_cypher": bool(cypher_result and "No results found" not in cypher_result),
            "used_vector": bool(vector_docs and len(vector_docs) > 0),
            "vector_docs_retrieved": len(vector_docs) if vector_docs else 0,
            
            # Answer Quality Indicators (heuristic-based)
            "contains_numbers": any(char.isdigit() for char in answer),
            "contains_specific_routes": any(code in answer for code in ["CAI", "HBE", "JFK", "LAX"]),
            "mentions_generation": any(gen in answer.lower() for gen in 
                                      ["millennial", "gen x", "gen z", "baby boomer", "generation"]),
            "answer_confidence": self.assess_confidence(answer),
            "answer_completeness": self.assess_completeness(answer, question),
        }
        
        # Relevance scoring
        if question in self.ground_truth:
            gt = self.ground_truth[question]
            metrics["expected_query_type"] = gt["query_type"]
            metrics["entity_coverage"] = self.calculate_entity_coverage(answer, gt["expected_entities"])
        
        self.results.append(metrics)
        return metrics
    
    def assess_confidence(self, answer):
        """Assess confidence level based on language used"""
        low_confidence_phrases = [
            "could not be found", "not found", "unable to", 
            "no data", "insufficient", "unclear"
        ]
        high_confidence_phrases = [
            "specifically", "exactly", "precisely", "according to",
            "the data shows", "results indicate"
        ]
        
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in low_confidence_phrases):
            return "low"
        elif any(phrase in answer_lower for phrase in high_confidence_phrases):
            return "high"
        else:
            return "medium"
    
    def assess_completeness(self, answer, question):
        """Assess if answer addresses all parts of the question"""
        question_keywords = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "for", "to", "from"}
        question_keywords -= stop_words
        
        overlap = len(question_keywords & answer_words) / len(question_keywords) if question_keywords else 0
        
        if overlap > 0.6:
            return "high"
        elif overlap > 0.3:
            return "medium"
        else:
            return "low"
    
    def calculate_entity_coverage(self, answer, expected_entities):
        """Calculate what percentage of expected entities appear in answer"""
        answer_lower = answer.lower()
        found = sum(1 for entity in expected_entities if entity.lower() in answer_lower)
        return found / len(expected_entities) if expected_entities else 0
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        # Aggregate by model
        model_stats = df.groupby('model').agg({
            'response_time_seconds': ['mean', 'std', 'min', 'max'],
            'answer_length_words': ['mean', 'std'],
            'vector_docs_retrieved': 'mean',
            'contains_numbers': 'sum',
            'contains_specific_routes': 'sum',
            'entity_coverage': 'mean'
        }).round(3)
        
        return {
            'detailed_results': df,
            'model_statistics': model_stats,
            'raw_results': self.results
        }

def create_comparison_visualizations(comparison_data):
    """Create interactive visualizations for model comparison"""
    
    df = comparison_data['detailed_results']
    
    # 1. Response Time Comparison
    fig_time = go.Figure()
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        fig_time.add_trace(go.Box(
            y=model_data['response_time_seconds'],
            name=model,
            boxmean='sd'
        ))
    fig_time.update_layout(
        title="Response Time Comparison Across Models",
        yaxis_title="Response Time (seconds)",
        xaxis_title="Model"
    )
    
    # 2. Answer Quality Metrics
    quality_metrics = df.groupby('model').agg({
        'answer_completeness': lambda x: (x == 'high').sum() / len(x),
        'answer_confidence': lambda x: (x == 'high').sum() / len(x),
        'entity_coverage': 'mean'
    }).reset_index()
    
    fig_quality = go.Figure()
    fig_quality.add_trace(go.Bar(
        name='Completeness (High %)',
        x=quality_metrics['model'],
        y=quality_metrics['answer_completeness'] * 100
    ))
    fig_quality.add_trace(go.Bar(
        name='Confidence (High %)',
        x=quality_metrics['model'],
        y=quality_metrics['answer_confidence'] * 100
    ))
    fig_quality.add_trace(go.Bar(
        name='Entity Coverage (%)',
        x=quality_metrics['model'],
        y=quality_metrics['entity_coverage'] * 100
    ))
    fig_quality.update_layout(
        title="Answer Quality Comparison",
        yaxis_title="Percentage (%)",
        xaxis_title="Model",
        barmode='group'
    )
    
    # 3. Context Usage Pattern
    context_usage = df.groupby('model').agg({
        'used_cypher': 'sum',
        'used_vector': 'sum',
        'vector_docs_retrieved': 'mean'
    }).reset_index()
    
    fig_context = go.Figure()
    fig_context.add_trace(go.Bar(
        name='Used Cypher Query',
        x=context_usage['model'],
        y=context_usage['used_cypher']
    ))
    fig_context.add_trace(go.Bar(
        name='Used Vector Store',
        x=context_usage['model'],
        y=context_usage['used_vector']
    ))
    fig_context.update_layout(
        title="Context Source Usage by Model",
        yaxis_title="Number of Times Used",
        xaxis_title="Model",
        barmode='group'
    )
    
    return {
        'response_time': fig_time,
        'quality_metrics': fig_quality,
        'context_usage': fig_context
    }

def run_comparison_suite(config, vectorstore, test_questions, models_to_compare, context_mode):
    """Run automated comparison across multiple models"""
    
    comparison = ModelComparison()
    results_summary = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tests = len(test_questions) * len(models_to_compare)
    current_test = 0
    
    for model_name in models_to_compare:
        st.subheader(f"Testing: {model_name}")
        model_data = MODEL_OPTIONS[model_name]
        
        # Initialize chain for this model
        with st.spinner(f"Initializing {model_name}..."):
            llm, retriever, qa_prompt = initialize_rag_chain(
                config, vectorstore, model_data, context_mode
            )
        
        if llm is None:
            st.error(f"Failed to initialize {model_name}")
            continue
        
        for question in test_questions:
            current_test += 1
            progress_bar.progress(current_test / total_tests)
            status_text.text(f"Testing {model_name} on question {current_test}/{total_tests}")
            
            # Get Cypher context
            start_time = time.time()
            format_result, intent = extract_intent_and_entities(question)
            
            # Get answer based on context mode
            try:
                if context_mode == "cypher_only":
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
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": question,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                    
                else:  # both
                    retrieved_docs = retriever.invoke(question)
                    vector_structured = format_retrieved_docs_to_structured(retrieved_docs)
                    
                    chain = qa_prompt | llm
                    result = chain.invoke({
                        "input": question,
                        "cypher_context": format_result,
                        "vector_context": vector_structured
                    })
                    answer = result.content if hasattr(result, 'content') else str(result)
                
                response_time = time.time() - start_time
                
                # Evaluate response
                metrics = comparison.evaluate_response(
                    question=question,
                    answer=answer,
                    model_name=model_name,
                    context_mode=context_mode,
                    response_time=response_time,
                    cypher_result=format_result,
                    vector_docs=retrieved_docs
                )
                
                results_summary.append({
                    'model': model_name,
                    'question': question[:50] + "...",
                    'time': f"{response_time:.2f}s",
                    'confidence': metrics['answer_confidence'],
                    'completeness': metrics['answer_completeness']
                })
                
            except Exception as e:
                st.error(f"Error testing {model_name} on question: {str(e)}")
                continue
    
    progress_bar.progress(1.0)
    status_text.text("Comparison complete!")
    
    return comparison, results_summary

def main():
    st.set_page_config(page_title="🔬 Multi-Model RAG Comparison", layout="wide")
    st.title("🔬 Multi-Model RAG Comparison System")
    st.markdown("Compare performance of different LLMs on airline query tasks")
    
    # Initialize components
    config, vectorstore = setup_static_rag_components()
    
    if config is None or vectorstore is None:
        st.error("Failed to initialize RAG components")
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("Comparison Configuration")
    
    # Model selection
    st.sidebar.subheader("Select Models to Compare")
    models_to_compare = []
    for model_name in MODEL_OPTIONS.keys():
        if st.sidebar.checkbox(model_name, value=True):
            models_to_compare.append(model_name)
    
    if len(models_to_compare) < 2:
        st.warning("Please select at least 2 models to compare")
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
            "Which generation of passengers reported the lowest food satisfaction?",
            "What is the average food satisfaction score for flights from CAI to HBE?",
            "Show me the top 3 routes by passenger satisfaction",
            "How many passengers from the Baby Boomer generation flew on route JFK to LAX?",
            "What routes have the highest average delay?"
        ]
    else:
        custom_questions = st.sidebar.text_area(
            "Enter questions (one per line):",
            height=200
        )
        test_questions = [q.strip() for q in custom_questions.split('\n') if q.strip()]
    
    st.sidebar.markdown(f"**Total Tests:** {len(test_questions) * len(models_to_compare)}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Run Comparison", "📈 Visualizations", "📋 Detailed Results"])
    
    with tab1:
        st.header("Run Automated Comparison")
        
        st.info(f"**Models Selected:** {', '.join(models_to_compare)}")
        st.info(f"**Context Mode:** {context_mode_name}")
        st.info(f"**Number of Test Questions:** {len(test_questions)}")
        
        with st.expander("View Test Questions"):
            for i, q in enumerate(test_questions, 1):
                st.write(f"{i}. {q}")
        
        if st.button("🚀 Start Comparison", type="primary"):
            comparison, results_summary = run_comparison_suite(
                config, vectorstore, test_questions, models_to_compare, context_mode
            )
            
            # Store in session state
            st.session_state['comparison'] = comparison
            st.session_state['results_summary'] = results_summary
            
            st.success("✅ Comparison completed!")
            
            # Quick summary table
            st.subheader("Quick Results Summary")
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.header("Comparison Visualizations")
        
        if 'comparison' in st.session_state:
            comparison_data = st.session_state['comparison'].generate_comparison_report()
            
            if comparison_data:
                visualizations = create_comparison_visualizations(comparison_data)
                
                st.plotly_chart(visualizations['response_time'], use_container_width=True)
                st.plotly_chart(visualizations['quality_metrics'], use_container_width=True)
                st.plotly_chart(visualizations['context_usage'], use_container_width=True)
                
                # Statistical summary
                st.subheader("Statistical Summary")
                st.dataframe(comparison_data['model_statistics'], use_container_width=True)
        else:
            st.info("Run a comparison first to see visualizations")
    
    with tab3:
        st.header("Detailed Results")
        
        if 'comparison' in st.session_state:
            comparison_data = st.session_state['comparison'].generate_comparison_report()
            
            if comparison_data:
                df = comparison_data['detailed_results']
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    selected_model = st.selectbox(
                        "Filter by Model:",
                        options=["All"] + list(df['model'].unique())
                    )
                with col2:
                    selected_confidence = st.selectbox(
                        "Filter by Confidence:",
                        options=["All", "high", "medium", "low"]
                    )
                
                # Apply filters
                filtered_df = df.copy()
                if selected_model != "All":
                    filtered_df = filtered_df[filtered_df['model'] == selected_model]
                if selected_confidence != "All":
                    filtered_df = filtered_df[filtered_df['answer_confidence'] == selected_confidence]
                
                # Display detailed results
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
                        col2.metric("Confidence", row['answer_confidence'])
                        col3.metric("Completeness", row['answer_completeness'])
                        col4.metric("Word Count", row['answer_length_words'])
        else:
            st.info("Run a comparison first to see detailed results")

if __name__ == "__main__":
    main()