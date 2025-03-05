import pandas as pd
import numpy as np
from tqdm import tqdm

fp = "CUB/bird_info_mini.csv"
#fp = "CUB/bird_info_mini.csv"
#df = pd.read_csv(fp)
#print(df.iloc[1])

# Sample small dataset of 3 species
data = {
    "Domain": ["Eukaryota", "Eukaryota", "Eukaryota"],
    "Kingdom": ["Animalia", "Animalia", "Animalia"],
    "Phylum": ["Chordata", "Chordata", "Chordata"],
    "Class": ["Aves", "Aves", "Aves"],
    "Order": ["Procellariiformes", "Procellariiformes", "Passeriformes"],
    "Family": ["Diomedeidae", "Diomedeidae", "Icteridae"],
    "Genus": ["Phoebastria", "Phoebastria", "Euphagus"],
    "Species": ["Black footed Albatross", "Laysan Albatross", "Brewer Blackbird"]
}

#df = pd.DataFrame(data)
df = pd.read_csv(fp)

def is_descendant(df, val1, val2):
    for _, row in df.iterrows(): #iterate through each row
        if val1 in row.values and val2 in row.values:
            val1_idx = row.index.get_loc(row[row == val1].index[0])  #trace up to compare the position
            val2_idx = row.index.get_loc(row[row == val2].index[0])
            if val1_idx is not None and val2_idx is not None and val1_idx < val2_idx:
                return True
    return False

# Function to convert CSV DataFrame into a matrix
def csv_2_matrix(df):
    unique_values = []
    for column_name in df.columns[-4:]:  # Focus on last 4 columns, from left to right
        uv_col_list = df[column_name].dropna().unique().tolist()  # Extract unique values
        unique_values += uv_col_list
    #return unique_values
    mat = []
    for i in tqdm(unique_values):
        for j in unique_values:
            if is_descendant(df, i, j): #if j is a descendant of i
                mat.append(1)
            else:
                mat.append(0)
    
    mat = np.array(mat).reshape(len(unique_values), len(unique_values))
    return mat

# Run the function on the small dataset
result_matrix = csv_2_matrix(df)
#print(result_matrix)
#np.savetxt('matrix.csv', result_matrix, delimiter=',')
np.save('cub_matrix_mini.npy', result_matrix)