from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDnVvYxGitJN1SiJvUHPSQsuHyb--kBUio"

INTENTS = [
    "Route_Satisfaction_Score",
    "Top_Routes_By_Flight_Count",
    "Flight_Count_By_Route",
    "Average_Food_Score_By_Flight",
    "Outgoing_Airports",
    "Passenger_Count_By_Generation_All",
    "Passenger_Count_By_Generation_Route",
    "Min_Delay_Flights_By_Route",
    "Max_Delay_Flights_By_Route",
    "Top_Routes_By_Delay",
    "Top_Routes_By_Satisfaction_Percentage",
    "Bottom_Routes_By_Satisfaction_Percentage",
    "Outgoing_Flights",
    "Satisfaction_By_Leg_Count",
    "Satisfaction_By_Loyalty_Level"
]

def build_intent_classifier(intents=INTENTS, model_name="models/gemini-2.5-flash"):
    template = (
        "You are the intent classification component of an airline assistant.\n\n"
        
        "## Important Instructions:\n"
        "1. Understand that users may refer to airports using:\n"
        "   - Full airport names (e.g., 'Cairo International Airport')\n"
        "   - City names (e.g., 'Cairo')\n"
        "   - Airport codes (e.g., 'CAI', 'JFK', 'cai', 'jfk')\n"
        "   - Variations with/without 'Airport' (e.g., 'Cairo Airport' vs 'Cairo')\n\n"
        
        "2. NORMALIZE all location references to UPPERCASE airport codes:\n"
        "   - 'Cairo' / 'Cairo Airport' / 'Cairo International' / 'cai' → 'CAI'\n"
        "   - 'Alexandria' / 'Borg El Arab' / 'hbe' → 'HBE'\n"
        "   - 'JFK Airport' / 'John F Kennedy' / 'jfk' → 'JFK'\n"
        "   - CRITICAL: All airport codes MUST be in UPPERCASE letters\n\n"
        
        "3. Handle common spelling mistakes gracefully:\n"
        "   - 'Cario' → 'Cairo' → 'CAI'\n"
        "   - 'Alexndria' → 'Alexandria' → 'HBE'\n\n"
        
        f"## Available Intents:\n{', '.join(intents)}\n\n"
        
        "## User Request:\n"
        "{spoken_request}\n\n"
        
        "## Response Format:\n"
        "You MUST respond ONLY with valid JSON (no markdown, no explanation).\n"
        "The arguments object MUST ALWAYS include these three keys with their values (use empty string \"\" if not applicable):\n"
        "{{\"intent\": \"<intent_name>\", \"arguments\": {{\"origin_station_code\": \"<CODE>\", \"destination_station_code\": \"<CODE>\", \"flight_number\": \"<NUMBER>\"}} }}\n\n"
        
        "CRITICAL RULES:\n"
        "- Always include all three keys: origin_station_code, destination_station_code, flight_number\n"
        "- Use empty string \"\" for any key that doesn't apply to the query\n"
        "- Airport codes MUST be UPPERCASE (CAI, not cai)\n"
        "- Flight numbers should be digits only\n\n"
        
        "Example responses:\n"
        "- User: 'flights from Cairo to Alexandria'\n"
        "  Response: {{\"intent\": \"Flight_Count_By_Route\", \"arguments\": {{\"origin_station_code\": \"CAI\", \"destination_station_code\": \"HBE\", \"flight_number\": \"\"}}}}\n\n"
        
        "- User: 'satisfaction score for JFK to LAX route'\n"
        "  Response: {{\"intent\": \"Route_Satisfaction_Score\", \"arguments\": {{\"origin_station_code\": \"JFK\", \"destination_station_code\": \"LAX\", \"flight_number\": \"\"}}}}\n\n"
        
        "- User: 'show me flight 123 details'\n"
        "  Response: {{\"intent\": \"Average_Food_Score_By_Flight\", \"arguments\": {{\"origin_station_code\": \"\", \"destination_station_code\": \"\", \"flight_number\": \"123\"}}}}\n\n"
        
        "- User: 'top routes by satisfaction'\n"
        "  Response: {{\"intent\": \"Top_Routes_By_Satisfaction_Percentage\", \"arguments\": {{\"origin_station_code\": \"\", \"destination_station_code\": \"\", \"flight_number\": \"\"}}}}\n"
    )

    prompt = PromptTemplate(
        input_variables=["spoken_request"],
        template=template,
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    chain = prompt | llm | (lambda x: x.content)
    return chain

def evaluate(chain, text):
    return chain.invoke({"spoken_request": text})