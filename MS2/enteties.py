from gliner import GLiNER

def merge_entities(entities, text):
    if not entities:
        return []
    merged = []
    current = entities[0]

    for next_entity in entities[1:]:
        if next_entity['label'] == current['label']: 
            current['end'] = next_entity['end']
            current['text'] = text[current['start']:current['end']]
        else:
            merged.append(current)
            current = next_entity

    merged.append(current)

    # Post-process: keep only digits for flight_number
    for ent in merged:
        if ent['label'] == 'flight_number':
            # extract only digits
            ent['text'] = ''.join(filter(str.isdigit, ent['text']))
    return merged


model = GLiNER.from_pretrained("numind/NuNerZero")
labels = ["origin_station_code", "destination_station_code", "flight_number"]
labels = [l.lower() for l in labels]
