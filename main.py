from Create_kg import Neo4jConnection
from run_queries import execute_intent
from classify import build_intent_classifier, evaluate
from enteties import model, merge_entities, labels
import json

def format_data_for_llm(data_list, intent):
    """
    Transforms a list of dictionaries into a single string of key:value pairs
    separated by commas, with each dictionary's output separated by a semicolon.
    The string ends with a period.

    Args:
        data_list (list): A list of dictionaries to process.

    Returns:
        str: The formatted string.
    """
    formatted_dicts = []
    
    # 1. Iterate through each dictionary in the list
    for data_dict in data_list:
        # 2. Convert key:value pairs into 'key:value' strings
        key_value_pairs = [f"{key}:{value}" for key, value in data_dict.items()]
        
        # 3. Join the pairs with a comma and space
        dict_string = ", ".join(key_value_pairs)
        
        formatted_dicts.append(dict_string)
        
    # 4. Join the formatted dictionary strings with a semicolon and space
    # 5. Append a period to the very end
    final_string = "; ".join(formatted_dicts) + "."
    final_string2 = f"The results of the query with intent '{intent}' are: {final_string}"
    
    return final_string2
def main():
    user_text = input("\nEnter your query: ")

    # 1️⃣ Classify intent
    chain = build_intent_classifier()
    llm_response = evaluate(chain, user_text)
    print("\n[LLM RAW OUTPUT]", llm_response)
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
        return

    intent = parsed.get("intent")
    args = parsed.get("arguments", {})
    print("\nDetected Intent:", intent)
    print("Arguments:", args)
    args={}

    # 2️⃣ Extract entities
    entities = model.predict_entities(user_text, labels)
    print(entities)
    entities = merge_entities(entities ,user_text)
    print("\nExtracted Entities:")
    for ent in entities:
        print(ent["text"], "=>", ent["label"])

    # 3️⃣ Connect to Neo4j
    config = Neo4jConnection.read_config("config.txt")
    neo = Neo4jConnection(config["NEO4J_URI"], config["NEO4J_USER"], config["NEO4J_PASSWORD"])

    # 4️⃣ Execute intent
    print("\nRunning Query...")
    result = execute_intent(intent, args, entities, neo)
    print("\nQuery Result:", result)
    format_result = format_data_for_llm(result,intent)
    print("\nFormatted Result:", format_result)

    neo.CloseConnection()
if __name__ == "__main__":
    main()
