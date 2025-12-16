def convert_entities_to_dict(entities):
    """
    Convert GLiNER entity list to a dict:
    [{"text": "LAX", "label": "origin_station_code"}] ->
    {"origin_station_code": "LAX"}
    """
    ent_dict = {}
    for ent in entities:
        ent_dict[ent["label"]] = ent["text"]
    return ent_dict

def execute_intent(intent, args, entities, neo):
    """
    Execute the correct Neo4j query based on the intent.
    Arguments come from the LLM.
    Entities come from GLiNER.

    Returns: A tuple (query_result, query_string)
    """

    # Convert entities list → dictionary
    ent = convert_entities_to_dict(entities)

    # Merge args + entities (entity overrides nothing)
    params = {**args, **ent}
    print(params)

    # Initialize query string
    query = None

    # ---------------------------
    #  Intent: Route_Satisfaction_Score
    # ---------------------------
    if intent == "Route_Satisfaction_Score":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport{station_code: $destination_station_code})
        MATCH (j: Journey)-[:ON]->(f)
        WHERE j.satisfaction_score > 3
        RETURN count(DISTINCT j) AS count_of_satisfied_passengers
        """
        return neo.run_query(query, params) , query

    # ---------------------------
    #  Intent: Top_Routes_By_Flight_Count
    # ---------------------------
    if intent == "Top_Routes_By_Flight_Count":
        query = """
        MATCH (a: Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(a2: Airport)
        RETURN a.station_code AS origin, a2.station_code AS destination, count(f) AS flight_count 
        ORDER BY flight_count DESC
        LIMIT 100
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Flight_Count_By_Route
    # ---------------------------
    if intent == "Flight_Count_By_Route":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport{station_code: $destination_station_code})
        RETURN count(DISTINCT f) AS count_of_flights
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Average_Food_Score_By_Flight
    # ---------------------------
    if intent == "Average_Food_Score_By_Flight":
        query = """
        MATCH (j: Journey)-[:ON]->(f: Flight{flight_number: $flight_number})
        RETURN avg(j.food_satisfaction_score) AS average_food_satisfaction
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Outgoing_Airports
    # ---------------------------
    if intent == "Outgoing_Airports":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport)
        RETURN d.station_code AS destination
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Passenger_Count_By_Generation_All
    # ---------------------------
    if intent == "Passenger_Count_By_Generation_All":
        query = """
        MATCH (p: Passenger)
        RETURN p.generation AS generation, count(p) AS count_of_passengers
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Passenger_Count_By_Generation_Route
    # ---------------------------
    if intent == "Passenger_Count_By_Generation_Route":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport{station_code: $destination_station_code})
        MATCH (p: Passenger)-[:TOOK]->(j: Journey)-[:ON]->(f)
        RETURN p.generation AS generation, count(DISTINCT p) AS count_of_passengers
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Min_Delay_Flights_By_Route
    # ---------------------------
    if intent == "Min_Delay_Flights_By_Route":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport{station_code: $destination_station_code})
        MATCH (j: Journey)-[:ON]->(f)
        RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type_description, avg(j.arrival_delay_minutes) AS average_delay
        ORDER BY average_delay ASC LIMIT 100
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Max_Delay_Flights_By_Route
    # ---------------------------
    if intent == "Max_Delay_Flights_By_Route":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport{station_code: $destination_station_code})
        MATCH (j: Journey)-[:ON]->(f)
        RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type_description, avg(j.arrival_delay_minutes) AS average_delay
        ORDER BY average_delay DESC LIMIT 100
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Top_Routes_By_Delay
    # ---------------------------
    if intent == "Top_Routes_By_Delay":
        query = """
        MATCH (o: Airport)<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport)
        MATCH (j: Journey)-[:ON]->(f)
        RETURN o.station_code AS origin, d.station_code AS destination, avg(j.arrival_delay_minutes) AS average_delay
        ORDER BY average_delay DESC LIMIT 100
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Top_Routes_By_Satisfaction_Percentage
    # ---------------------------
    if intent == "Top_Routes_By_Satisfaction_Percentage":
        query = """
        MATCH (o: Airport)<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport)
        MATCH (j: Journey)-[:ON]->(f)
        RETURN o.station_code AS origin, 
            d.station_code AS destination, 
            count(DISTINCT CASE WHEN j.satisfaction_score > 3 THEN j END) AS count_of_satisfied_passengers,
            100.0 * count(DISTINCT CASE WHEN j.satisfaction_score > 3 THEN j END) / count(DISTINCT j) AS percentage_of_satisfied_passenegers
        ORDER BY count_of_satisfied_passengers DESC 
        LIMIT 100
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Bottom_Routes_By_Satisfaction_Percentage
    # ---------------------------
    if intent == "Bottom_Routes_By_Satisfaction_Percentage":
        query = """
        MATCH (o: Airport)<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport)
        MATCH (j: Journey)-[:ON]->(f)
        RETURN o.station_code AS origin, 
            d.station_code AS destination, 
            count(DISTINCT CASE WHEN j.satisfaction_score <= 3 THEN j END) AS count_of_dissatisfied_passengers,
            100.0 * count(DISTINCT CASE WHEN j.satisfaction_score <= 3 THEN j END) / count(DISTINCT j) AS percentage_of_dissatisfied_passenegers
        ORDER BY count_of_dissatisfied_passengers DESC 
        LIMIT 100
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Outgoing_Flights
    # ---------------------------
    if intent == "Outgoing_Flights":
        query = """
        MATCH (o: Airport{station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f: Flight)-[:ARRIVES_AT]->(d: Airport)
        RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type_description, d.station_code as destination
        """
        return neo.run_query(query, params), query

    # ---------------------------
    #  Intent: Satisfaction_By_Leg_Count
    # ---------------------------
    if intent == "Satisfaction_By_Leg_Count":
        query = """
        MATCH (j: Journey)
        RETURN j.number_of_legs as number_of_legs, avg(j.satisfaction_score) AS average_satisfaction_score
        ORDER BY number_of_legs ASC
        """
        return neo.run_query(query), query

    # ---------------------------
    #  Intent: Satisfaction_By_Loyalty_Level
    # ---------------------------
    if intent == "Satisfaction_By_Loyalty_Level":
        query = """
        MATCH (p: Passenger)-[:TOOK]->(j: Journey)
        RETURN p.loyalty_program_level AS loyalty_program_level, avg(j.satisfaction_score) AS average_satisfaction_score
        """
        return neo.run_query(query), query

    return {"error": "Unknown intent"}
