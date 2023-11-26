from genEmbYesLabeled import yesLabeledEmbedding
import pandas as pd
import numpy as np
import os 
import random as rand

excel_file_path = 'output.xlsx'

def writeToExcel(embeddings):
    df = pd.DataFrame(embeddings, columns=[f'Column{i}' for i in range(1, len(embeddings[0])+1)])
    if os.path.exists(excel_file_path):
        # İkinci DataFrame'i dosyaya ekleyerek yaz
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, index=False, header=False)
    else:
        df.to_excel(excel_file_path, index=False)

TOTAL_NUMBER_OF_YES_LABELED = 1000
TOTAL_NUMBER_OF_NO_LABELED = 1000

# random sayi sayesinde bunlardan bir tane her seferinde
# olusturacagiz

TOTAL_NUMBER_OF_DIFFERENT_GRAPH_GEN = 0
def generateData():
    df=[]
    i = 0
    yes_temp = TOTAL_NUMBER_OF_YES_LABELED
    no_temp = TOTAL_NUMBER_OF_NO_LABELED
    for i in range(4):
        # Burada olabilecek tum graph cesitlerini koyup ayri ayri deneyecegiz"
        graphEmbedding = yesLabeledEmbedding(totalNumberOfNodes= 100, edgesBetweenPartitions=3)
        if i == 0:
            df = graphEmbedding
        else : 
            df = np.vstack((df, graphEmbedding))
    writeToExcel(df)

generateData()
