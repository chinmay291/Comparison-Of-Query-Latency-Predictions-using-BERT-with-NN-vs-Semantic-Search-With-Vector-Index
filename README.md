# dbsi-project1
SQL Query Execution Time Prediction using Semantic Search(using LangChain and Llama Index)

Query optimization is a vast and actively researched field within databases, and if we could accurately predict the execution time of a query without looking at the query plan, this information could be fed back to the query optimizer to improve its results. The first step in doing so would be to be able to predict SQL query execution time. There are several approaches to this that can be found in the literature, but leveraging recent advancements in generative AI is a new subtopic that the community has developed interest in recently. 

The goal of this project is to predict SQL Query Execution times using recent advancements in Generative AI. More specifically, we use semantic search on PostgreSQL query plans to identify the plans that are most similar to a given query. 

The first step is to collect data that will be used to create the vector embeddings. While there are a number of ways to do this in PostgreSQL, auto_explain was the most attractive one since I wanted to extract queries, query plans and query durations for a large number of queries. auto_explain is a shared loadable library that ships with PostgreSQL source code and allows the duration of each executed query and its query plan to be logged. This is enabled by setting the following configuration parameters:
shared_preload_libraries = 'auto_explain'       
auto_explain.log_min_duration = 0
auto_explain.log_format = JSON
log_destination = 'stderr,csvlog'
logging_collector = on 

The next step was to create a sample dataset, run some queries and let PostgreSQL generate the auto_explain logs. I picked the standard pgbench benchmark for this purpose. I generated data with a scale factor of 10,000 and executed a balanced read/write workload benchmark. The logs generated from this execution were used for the next step.

The next part can be summarized as follows:
1. Preprocessing the generated log file to produce the dataset in its required form.
2. Pick the correct embeddings to use. I used the default all-MiniLM-L6-v2 embeddings that LangChain provides. 
3. Split the generated data into two parts: 80% would be used to create the vector index and 20% would be used as test data.
4. Create a GPTVectorStoreIndex(provided by Llama Index) using 80% of the query execution data.
5. Query the index using a retriever and find the K most similar vectors matching a query. Choose the average of the execution times of the K data points as the predicted value. 

With this rudimentary semantic search model, I was able to achieve a R Squared Error(Coefficient of Determination) of 0.54. This number can serve as a baseline as I look for more effective ways to do this during the remaining part of the semester. This problem by nature is not an easy prediction problem as it is hard to predict the query execution time without actually executing the query. The state of the various internal components of the database such as the buffer cache and the storage manager is determined not just by an individual query but by the workload’s access patterns. Exposing the model to some of these components and their interactions many help in improving our results in the future.  


The file 'query_latency_semantic_search.py' contains the main code used for processing raw log files and creating a vector search index and testing its accuracy. 
The postgresql_config directory contains the PostgreSQL configuration file for the instance of Postgres that was used for data collection. 
A sample raw log output file is present in the sample_log_file directory.

All the dependencies for the projects are mentioned in requirements.txt

Future Work:

There are several directions which can be explored as I continue building on this work:

1. Postgres query plans are actually collected as JSON documents, but the current model treats them as unstructured text. Creating a structure-aware vector index which understands and learns from the semi-structured format of JSON documents can provide better search results. 

2. Tune the semantic search workflow by trying different embeddings, index and retriever configurations. 

3. Instead of treating this as a semantic search problem, we can use BERT to actually learn the representation of queries and create a better vectorized representation than using the pretrained embeddings that LLMs use.

4. Integrate these capabilities into EvaDB. The entire workflow right now is standalone and operates outside of EvaDB, but if we can develop an accurate query execution time prediction model, it can be integrated with EvaDB.  
