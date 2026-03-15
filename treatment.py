import json

with open("treatment.json") as f:
    treatments = json.load(f)

def get_treatment(disease):

    if disease in treatments:
        return treatments[disease]

    return "Consult a dermatologist."