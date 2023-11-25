import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statistics
from sentence_transformers import SentenceTransformer
import xgboost as xg 
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from llama_index import SimpleDirectoryReader

file1 = open("./postgresql-2023-10-17_160558.csv", 'r')
count = 0

durations = []
queries_and_plans = []
 
while True:
    count += 1
 
    # Get next line from file
    line = file1.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break
#     print("Line{}: {}".format(count, line.strip()))
    if(line[0:5] == "2023-"):
#         print(line)
        components = line.split(",")
#         print(components[-1])
        subcomponents = components[-1].split()
#         print(subcomponents)
        if subcomponents[0] == "\"duration:":
#             print(float(subcomponents[1]))
            durations.append(float(subcomponents[1]))
    elif line[0] == "{":
        json_string = ""
        json_string += line
        while True:
            json_line = file1.readline()
            if json_line[0] == "}":
                json_string += "}\n"
                break
            else:
                json_string += json_line
        queries_and_plans.append(json_string)
            
 
file1.close()

print(statistics.mean(durations))
print(statistics.stdev(durations))

num_files = len(queries_and_plans)
for i in range(num_files):
    text_file = open("extracted_json_files/" + str(i) + ".txt", "w")
    text_file.write(queries_and_plans[i])
    text_file.write("\n")
    text_file.write(str(durations[i]))
    text_file.close()

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def extractQuery(document):
    doc_split = document.split("\n")
    return doc_split[1]

def extractExecTime(document):
    doc_split = document.split("\n")
    return float(doc_split[-1])


documents = SimpleDirectoryReader("extracted_json_files").load_data()

queries = []
exec_times = []
for doc in documents:
        queries.append(extractQuery(doc.text))
        exec_times.append(extractExecTime(doc.text))

query_embeddings = bert_model.encode(queries)
query_plan_embeddings = bert_model.encode(queries_and_plans)



X = np.append(query_embeddings,query_plan_embeddings, axis=1)
y = np.array(exec_times)

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=7)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


svr = SVR(kernel = 'poly')
svr.fit(X_train, Y_train)

Y_pred = svr.predict(X_test)

coefficient_of_dermination = r2_score(Y_test, Y_pred)
print(coefficient_of_dermination)


xgb_r = xg.XGBRegressor(n_estimators = 10, seed = 123) 
xgb_r.fit(X_train, Y_train) 

Y_pred = xgb_r.predict(X_test) 

coefficient_of_dermination = r2_score(Y_test, Y_pred)
print(coefficient_of_dermination)

model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),   
        layers.Dense(64, activation='relu'), 
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'), 
        layers.Dense(1)
    ])
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=100, validation_split=0.1, batch_size=5)

Y_pred = model.predict(X_test)

coefficient_of_dermination = r2_score(Y_test, Y_pred)
print(coefficient_of_dermination)