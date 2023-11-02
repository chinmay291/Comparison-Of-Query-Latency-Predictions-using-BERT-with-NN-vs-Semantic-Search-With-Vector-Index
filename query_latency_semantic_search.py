import statistics
import math  
import sklearn.metrics 
from sklearn.metrics import r2_score 
from langchain.embeddings import SentenceTransformerEmbeddings
from llama_index import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex

def extractQuery(document):
    doc_split = document.split("\n")
    return doc_split[1]

def extractExecTime(document):
    doc_split = document.split("\n")
    return float(doc_split[-1])

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
# print(durations)

num_files = len(queries_and_plans)
for i in range(num_files):
    text_file = open("extracted_json_files/" + str(i) + ".txt", "w")
    text_file.write(queries_and_plans[i])
    text_file.write("\n")
    text_file.write(str(durations[i]))
    text_file.close()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

documents = SimpleDirectoryReader("extracted_json_files").load_data()

index = GPTVectorStoreIndex([])
counter = 0
test_queries = []
test_real_exec_times = []
for doc in documents:
    counter += 1
    if counter % 5 != 0:
        index.insert(doc)
    else:
        test_queries.append(extractQuery(doc.text))
        test_real_exec_times.append(extractExecTime(doc.text))

retriever = index.as_retriever()

test_predicted_exec_times = []
for i in range(len(test_queries)):
    query = test_queries[i]
    nodes = retriever.retrieve(query)
    predicted_exec_time_sum = 0
    for node in nodes:
        predicted_exec_time_sum += extractExecTime(node.text)
    predicted_exec_time_sum = predicted_exec_time_sum / len(nodes)
    test_predicted_exec_times.append(predicted_exec_time_sum)
mse = sklearn.metrics.mean_squared_error(test_real_exec_times, test_predicted_exec_times) 
print(mse)

rmse = math.sqrt(mse)
print(rmse)

coefficient_of_dermination = r2_score(test_real_exec_times, test_predicted_exec_times)
print(coefficient_of_dermination)






