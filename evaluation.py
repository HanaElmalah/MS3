# --- evaluation.py ---
import pandas as pd
import time
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import necessary functions/classes from your main file
from embeddings import ( 
    Neo4j, initialize_rag_chain, extract_intent_and_entities, 
    MODEL_OPTIONS, PREDEFINED_QUESTIONS, format_data_for_llm, 
    CONTEXT_OPTIONS, setup_static_rag_components
) 
# NOTE: Replace 'your_main_file' with the actual name of your file (e.g., 'rag_app')

# ----------------------------------------------------------------------
# 1. GROUND TRUTH GENERATION
# ----------------------------------------------------------------------

def generate_ground_truth(questions, config, vectorstore):
    """
    Runs Cypher-only queries to establish ground truth data (Cypher result + expected answer).
    """
    test_set_results = []
    
    # We must use one LLM (e.g., Gemini) to convert the *Cypher result* into a natural language *answer*.
    # This answer is our gold standard.
    # Set up the RAG chain for the ground truth generator (e.g., Gemini)
    llm_gt, _, qa_prompt_gt = initialize_rag_chain(
        config, 
        vectorstore, 
        MODEL_OPTIONS["Mistral 7B Instruct"], 
        CONTEXT_OPTIONS["Cypher Query Only"] # <-- ONLY CYPHER
    )
    if llm_gt is None:
        raise Exception("Failed to initialize Ground Truth LLM.")
        
    print("\n--- Generating Ground Truth Answers (Cypher Baseline) ---")
    
    for i, question in enumerate(questions):
        if not question.strip():
            continue
            
        print(f"[{i+1}/{len(questions)}] Processing: {question}")
        
        # 1. Get Cypher Result (The raw data/facts)
        # Note: You need to make sure extract_intent_and_entities is robust outside of Streamlit
        format_result, intent, query_itself, _ = extract_intent_and_entities(question)

        # 2. Get LLM-Formatted Answer (The Gold Standard Answer)
        # We use a simple LCEL chain for the Cypher-only mode: Prompt | LLM
        if format_result.startswith("No results found"):
             gt_answer = "The specific data needed to answer this question could not be found in the query results."
        else:
            chain = qa_prompt_gt | llm_gt
            result = chain.invoke({"input": question, "cypher_context": format_result})
            gt_answer = result.content if hasattr(result, "content") else str(result)
            
        test_set_results.append({
            "Question": question,
            "Intent": intent,
            "Cypher_Query": query_itself,
            "Cypher_Context": format_result,
            "Ground_Truth_Answer": gt_answer.strip()
        })
        
    return pd.DataFrame(test_set_results)


# ----------------------------------------------------------------------
# 2. RUNNING THE EXPERIMENTS
# ----------------------------------------------------------------------

def run_evaluation_experiments(df_gt, config, vectorstore):
    """
    Runs each question through each LLM and records the generated answer and latency.
    """
    experiment_data = []
    
    for model_name, model_data in MODEL_OPTIONS.items():
        print(f"\n--- Running Experiments for Model: {model_name} ---")
        
        # Initialize RAG chain for the current model in Cypher-Only mode
        # You will be using the Cypher_Context generated from the Ground Truth step
        llm, _, qa_prompt = initialize_rag_chain(
            config, 
            vectorstore, 
            model_data, 
            CONTEXT_OPTIONS["Cypher Query Only"] # <-- Keep this mode for consistency check
        )
        if llm is None:
            print(f"Skipping {model_name} due to initialization error.")
            continue

        for index, row in df_gt.iterrows():
            question = row['Question']
            cypher_context = row['Cypher_Context']
            
            # Run the query multiple times (e.g., 5-10 runs for consistency)
            NUM_CONSISTENCY_RUNS = 5 
            
            for run in range(1, NUM_CONSISTENCY_RUNS + 1):
                print(f"  > {model_name} | Q{index+1} | Run {run}")
                
                start_time = time.time()
                try:
                    # Execute the Cypher-Only Chain (Prompt | LLM)
                    chain = qa_prompt | llm
                    result = chain.invoke({"input": question, "cypher_context": cypher_context})
                    generated_answer = result.content if hasattr(result, "content") else str(result)
                except Exception as e:
                    generated_answer = f"ERROR: {e}"
                
                latency = time.time() - start_time
                
                experiment_data.append({
                    "Model": model_name,
                    "Question": question,
                    "Run": run,
                    "Latency_s": latency,
                    "Generated_Answer": generated_answer.strip(),
                    "Ground_Truth_Answer": row['Ground_Truth_Answer'], # Pull from GT DataFrame
                    "Cypher_Context": cypher_context
                })
                
    return pd.DataFrame(experiment_data)


# ----------------------------------------------------------------------
# 3. METRICS CALCULATION (Semantic Similarity)
# ----------------------------------------------------------------------

def calculate_metrics(df_results, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Calculates semantic similarity (consistency and accuracy) and correctness.
    """
    # Initialize the embedding model (use the same one as your RAG setup for semantic consistency)
    print(f"\n--- Calculating Metrics using {embedding_model_name} ---")
    embedder = SentenceTransformer(embedding_model_name)

    # 1. Calculate ACCURACY (Similarity to Ground Truth)
    # The gold standard for accuracy is the Ground_Truth_Answer
    # You will compare each Generated_Answer to the Ground_Truth_Answer
    
    # Generate embeddings for Ground Truth and Generated Answers
    gt_answers = df_results['Ground_Truth_Answer'].tolist()
    gen_answers = df_results['Generated_Answer'].tolist()
    
    gt_embeddings = embedder.encode(gt_answers, show_progress_bar=True)
    gen_embeddings = embedder.encode(gen_answers, show_progress_bar=True)
    
    # Calculate cosine similarity between each pair (Generated vs. Ground Truth)
    # This requires a row-by-row comparison, so we only need the diagonal of the similarity matrix
    similarity_scores = [cosine_similarity(
        gt_embeddings[i].reshape(1, -1), 
        gen_embeddings[i].reshape(1, -1)
    )[0][0] for i in range(len(df_results))]

    df_results['Accuracy_Similarity'] = similarity_scores
    
    # Apply a threshold for binary Correctness
    SIMILARITY_THRESHOLD = 0.85 # Tweak this value based on your models' output quality
    df_results['Is_Correct'] = df_results['Accuracy_Similarity'] >= SIMILARITY_THRESHOLD

    
    # 2. Calculate CONSISTENCY (Run-to-Run Similarity)
    # This is more complex. For each Question/Model group, you must compare all Run pairs.
    consistency_scores = []
    
    for (model, question), group in df_results.groupby(['Model', 'Question']):
        
        group_answers = group['Generated_Answer'].tolist()
        group_embeddings = embedder.encode(group_answers)
        
        # Calculate pairwise cosine similarity for all runs for this question/model
        pairwise_sim = cosine_similarity(group_embeddings)
        
        # Extract upper triangle (excluding diagonal) to get unique pairs
        # Calculate the average similarity score for the group
        upper_triangle_indices = np.triu_indices(pairwise_sim.shape[0], k=1)
        avg_pairwise_sim = np.mean(pairwise_sim[upper_triangle_indices])
        
        # Apply the average to all rows in the group
        consistency_scores.extend([avg_pairwise_sim] * len(group))
        
    df_results['Consistency_Score'] = consistency_scores


# ----------------------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------------------

def main_evaluation():
    # 1. Setup static components
    config, vectorstore = setup_static_rag_components() 
    if config is None:
         print("Setup failed. Exiting.")
         return

    # Filter questions known to be good for baseline for the GT generation
    questions_for_gt = [q for q in PREDEFINED_QUESTIONS if q.strip()]

    # 2. Generate Ground Truth
    df_gt = generate_ground_truth(questions_for_gt, config, vectorstore)
    
    # 3. Run Experiments
    df_results = run_evaluation_experiments(df_gt, config, vectorstore)
    
    # 4. Calculate Metrics
    # Assumes you use 'all-MiniLM-L6-v2' as in your setup_static_rag_components
    df_results_with_metrics = calculate_metrics(df_results, "all-MiniLM-L6-v2") 
    
    # 5. Save and Summarize
    # Save the detailed run-by-run results
    df_results_with_metrics.to_csv("llm_evaluation_details.csv", index=False)
    print("\nDetailed evaluation results saved to 'llm_evaluation_details.csv'")

    # Calculate the final summary table (The 'Accuracy Matrix')
    summary = df_results_with_metrics.groupby('Model').agg(
        Average_Accuracy_Sim=('Accuracy_Similarity', 'mean'),
        Binary_Accuracy=('Is_Correct', 'mean'),
        Average_Consistency_Sim=('Consistency_Score', 'mean'),
        Average_Latency_s=('Latency_s', 'mean'),
        Total_Runs=('Run', 'size')
    ).reset_index()
    
    summary['Binary_Accuracy'] = (summary['Binary_Accuracy'] * 100).round(2).astype(str) + '%'
    summary['Average_Accuracy_Sim'] = summary['Average_Accuracy_Sim'].round(4)
    summary['Average_Consistency_Sim'] = summary['Average_Consistency_Sim'].round(4)
    summary['Average_Latency_s'] = summary['Average_Latency_s'].round(2)


    summary.to_csv("llm_accuracy_matrix_summary.csv", index=False)
    print("\n--- Final Accuracy Matrix Summary ---")
    print(summary)

if __name__ == "__main__":
    # Ensure this part is only run when the script is executed directly
    # You will need to uncomment this when you are ready to run the batch process
    main_evaluation()
    # pass # Keeping it as pass for now, so I don't error out on the import

# --- END evaluation.py ---