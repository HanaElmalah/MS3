import csv
from neo4j import GraphDatabase, exceptions
import time

CSV_FILEPATH = "Airline_surveys_sample.csv"

class Neo4jConnection:
    def read_config(filepath):
        config = {}
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    key, value = line.strip().split('=')
                    config[key] = value.strip().strip('"')
            # print(config)
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
            print("Connection failed.")
            raise
        except Exception as e:
            print(f"Error occurred: {e}")
            raise

    def CloseConnection(self):
        if self.driver:
            self.driver.close()
            print("Connection closed.")
    def run_query(self, query, params=None):
        """
        Execute a Cypher query and return a list of dict results.
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"[QUERY ERROR] {e}")
            return []

    def CreateDB(self, tx):

        print("Setting up constraints for Airline Schema...")
        tx.run("CREATE CONSTRAINT passenger_id IF NOT EXISTS FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE")
        tx.run("CREATE CONSTRAINT journey_id IF NOT EXISTS FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE")
        tx.run("CREATE CONSTRAINT flight_composite_id IF NOT EXISTS FOR (f:Flight) REQUIRE (f.flight_number, f.fleet_type_description) IS UNIQUE")
        tx.run("CREATE CONSTRAINT airport_code IF NOT EXISTS FOR (a:Airport) REQUIRE a.station_code IS UNIQUE")

    def RowImport(self, tx, row):
        cypher_query = """
        // 1. Create/Merge Passenger node
        MERGE (p:Passenger {record_locator: $RecordLocator})
        ON CREATE SET 
            p.loyalty_program_level = $LoyaltyLevel, 
            p.generation = $Generation
        
        // 2. Create/Merge Journey node with all properties (scores/delays are integers)
        MERGE (j:Journey {feedback_ID: $FeedbackID})
        ON CREATE SET 
            j.food_satisfaction_score = toInteger($FoodScore),
            j.arrival_delay_minutes = toInteger($DelayMinutes),
            j.actual_flown_miles = toInteger($FlownMiles),
            j.number_of_legs = toInteger($NumberOfLegs),
            j.passenger_class = $Class
            
        // 3. Create/Merge Flight node (composite key)
        MERGE (f:Flight {
            flight_number: $FlightNumber, 
            fleet_type_description: $FleetType
        })

        // 4. Create/Merge Departure Airport node
        MERGE (dep:Airport {station_code: $DepartureAirport})

        // 5. Create/Merge Arrival Airport node
        MERGE (arr:Airport {station_code: $ArrivalAirport})

        // 6. Relationships creation based on the schema:
        // (Passenger)-[:TOOK]->(Journey)
        MERGE (p)-[:TOOK]->(j)
        
        // (Journey)-[:ON]->(Flight)
        MERGE (j)-[:ON]->(f)
        
        // (Flight)-[:DEPARTS_FROM]->(Airport)
        MERGE (f)-[:DEPARTS_FROM]->(dep)
        
        // (Flight)-[:ARRIVES_AT]->(Airport)
        MERGE (f)-[:ARRIVES_AT]->(arr)
        """
        
        tx.run(cypher_query, 
               RecordLocator=row['record_locator'], 
               LoyaltyLevel=row['loyalty_program_level'], 
               Generation=row['generation'],
               FeedbackID=row['feedback_ID'],
               FoodScore=row['food_satisfaction_score'],
               DelayMinutes=row['arrival_delay_minutes'],
               FlownMiles=row['actual_flown_miles'],
               NumberOfLegs=row['number_of_legs'],
               Class=row['passenger_class'],
               FlightNumber=row['flight_number'],
               FleetType=row['fleet_type_description'],
               DepartureAirport=row['origin_station_code'], 
               ArrivalAirport=row['destination_station_code'] 
        )
        
    def read_csv(self, csv_filepath):
        start_time = time.time()
        imported_count = 0
        try:
            with open(csv_filepath, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data_rows = list(reader)
                
                if not data_rows:
                    print("CSV file is empty.")
                    return


                with self.driver.session() as session:
                    session.execute_write(self.CreateDB)

                with self.driver.session() as session:
                    for i, row in enumerate(data_rows):
                        session.execute_write(self.RowImport, row)
                        imported_count += 1
                        
                        if (i + 1) % 100 == 0:
                            print(f"Processed {i + 1} records...")
                    
                end_time = time.time()
            
        except FileNotFoundError:
            print("CSV file not found.")
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    importer = None
    try:
        config = Neo4jConnection.read_config('config.txt')
        uri = config['NEO4J_URI']
        user = config['NEO4J_USER']
        password = config['NEO4J_PASSWORD']
        importer = Neo4jConnection(uri, user, password)
        importer.read_csv(CSV_FILEPATH)
    except Exception:
        pass
    finally:
        if importer:
            importer.CloseConnection()