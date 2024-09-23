# Banking Bot - End-to-End Implementation 

This project features a sophisticated banking bot developed using Large Language Models (LLM) and Retrieval-Augmented Generation (RAG). The bot is designed to assist users with various banking tasks, providing accurate and efficient responses by leveraging advanced natural language processing techniques and real-time data retrieval.

## To create a conda environment

`conda env create -f deploy\env.yml`

## To create the vector database for the word embeddings, run the following command

` python create_database.py`

## To run the chatbot, follow the below commands

`python app.py`

### If you running localy, try using the chatbot.py file. It has sentiment analysis and intent recognition additinally

`python chatbot.py`

Note : Add your OPENAI_API_KEY environment variable in your system before running the script.

