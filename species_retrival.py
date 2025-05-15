import pandas as pd
import requests
from sklearn.metrics import f1_score
import os 
from tqdm import tqdm
import argparse

ALLOWED_SPECIES = ['Homo sapiens', 'Mus musculus', 'Rattus norvegicus']
UNIPROT_TO_ALLOWED = {
    'Homo sapiens (Human)': 'Homo sapiens',
    'Mus musculus (Mouse)': 'Mus musculus',
    'Rattus norvegicus (Rat)': 'Rattus norvegicus'
}

def calculate_f1_score(results, labels):
    y_true = []
    y_pred = []
    for result, label in zip(results, labels):
        if result is None:
            y_pred.append(0)
        else:
            # different match 
            y_pred.append(1 if label in result else 0)
            #y_pred.append(1 if result is not None and result[0] == label and result[1] == label else 0)
        y_true.append(1)  
    return f1_score(y_true, y_pred)

def get_uniprot_species(gene_name):
    """
    use the uniprot api to get the species of the gene
    """
    url = f"https://rest.uniprot.org/uniprotkb/search?query={gene_name}&fields=organism_name&format=tsv&size=1"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            species_full = lines[1].strip()
            species = UNIPROT_TO_ALLOWED.get(species_full, None)
            return species
    return None

def get_species_mygene(gene_symbol):
    url = f"https://mygene.info/v3/query?q=symbol:{gene_symbol}&species=all&fields=species"
    r = requests.get(url)
    if r.status_code == 200:
        hits = r.json().get('hits', [])
        species_set = set()
        for hit in hits:
            species = hit.get('species')
            if isinstance(species, dict):
                name = species.get('name')
                if name:
                    species_set.add(name)
            elif isinstance(species, list):
                for s in species:
                    name = s.get('name')
                    if name:
                        species_set.add(name)
        return list(species_set)
    return []

def main():
    parser = argparse.ArgumentParser(description='species retrieval from entity names')
    parser.add_argument('--input', '-i', type=str, help='Path to input Excel file')
    parser.add_argument('--output', '-o', type=str, help='Path to output Excel file')
    args = parser.parse_args()

    file_path = args.input
    output_path = args.output
    df = pd.read_excel(file_path)  

    df = df[df['species'].notna() & (df['species'] != '')]
    df = df[df['species'].isin(ALLOWED_SPECIES)]

    results = []
    labels = []
    returned_species = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        subj = row['subj']
        obj = row['obj']
        label = row['species']
        
        subj_species = get_uniprot_species(subj)
        obj_species = get_uniprot_species(obj)

        if subj_species is None:
            subj_species_list = get_species_mygene(subj)
            subj_species = next((s for s in subj_species_list if s in ALLOWED_SPECIES), None)
        if obj_species is None:
            obj_species_list = get_species_mygene(obj)
            obj_species = next((s for s in obj_species_list if s in ALLOWED_SPECIES), None)
        
        df.loc[idx, 'subj_species'] = subj_species
        df.loc[idx, 'obj_species'] = obj_species

        if subj_species in ALLOWED_SPECIES and obj_species in ALLOWED_SPECIES:
            results.append((subj_species, obj_species))
            if subj_species == label or obj_species == label:
                df.loc[idx, 'correct'] = 1
            else:
                df.loc[idx, 'correct'] = 0
        elif subj_species in ALLOWED_SPECIES:
            results.append((subj_species, subj_species))
            if subj_species == label:
                df.loc[idx, 'correct'] = 1
            else:
                df.loc[idx, 'correct'] = 0
        elif obj_species in ALLOWED_SPECIES:
            results.append((obj_species, obj_species))
            if obj_species == label:
                df.loc[idx, 'correct'] = 1
            else:
                df.loc[idx, 'correct'] = 0
        else:
            results.append(None)
            df.loc[idx, 'corect'] = 0
            returned_species.add(subj_species)
            returned_species.add(obj_species)
        labels.append(label)
    
    # Count the number of None in results
    none_count = results.count(None)
    print(f"Number of interactions that cannot be retrieved: {none_count}")

    # calculate the f1 score
    f1 = calculate_f1_score(results, labels)
    print(f"F1 score: {f1:.4f}")

    # save the results
    df.to_excel(output_path, index=False)

    # save the returned species
    print(f"Number of returned species: {len(returned_species)}")
    print(f"Returned species: {returned_species}")

if __name__ == "__main__":
    main()
