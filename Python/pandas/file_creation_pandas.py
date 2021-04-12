'''
How to use pandas to save in a same file results from different moments in a code
Format .csv, maybe others?

If pandas is built upon numpy, could we handle .npy files with pandas?
'''

import pandas as pd
import webbrowser   # To display data in a web window for more clarity

df = pd.read_csv('survey_results_public.csv')
schema_df = pd.read_csv('survey_results_schema.csv')

# set options to cap the maximum rows and columns to show.
# Important for data with large number of rows and columns
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)

# head montre par défaut les 5 premières rangées, mais on peut spécifier
# La quantité qu'on souhaite voir. head pour entête ou un aperçu
# print(df.head()) # On doit le print() pour afficher dans VSC vs mettons jupyter

# print(df.columns)
# print(df.iloc[0:2, 0:2]) # iloc pour integer-location. fonction pour choisir dans les données
# selon un indice entier. il y a aussi DataFrame.loc qui permet de choisir selon un string
# correspondant au titre par exemple d'une colonne ou d'une rangée
example_dict = {
    'first' : ['Corey', 'Jane', 'John'],
    'last': ['Schafer', 'Doe', 'Doe'],
    'email': ['CoreyMSchafer@gmail.com', 'JaneDoe@gmail.com', 'JohnDoe@gmail.com']
}

# print(df.set_index('email')) # Montre le df avec la colonne des email servant d'indice des rangées plutôt
# que les entiers de 0 jusqu'à n quand on ne spécifie rien. Ne change pas le df, juste formatte l'affichage

df_example = pd.DataFrame()
df_example.to_csv('example.csv')
df_example
