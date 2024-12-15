from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import os
from RealtimeSTT import AudioToTextRecorder
import pyautogui

os.environ["OPENAI_API_KEY"] =os.environ.get('OPENAI_KEY')
pc = Pinecone(api_key=os.environ.get('PINECONE_KEY'))
index = pc.Index("sapp2")
client = OpenAI()
polling = True
prompt="You are a sales rep for a particular Solar panel installation agency. You will initiate a sales conversation with a customer. Speak like a sales agent actually would on the phone. Only use company information given to you alongside the question by the user. Be persuasive but simple in your responses. The goal is to be personalable and helpful to the client in order to sell the companies product"
conversation_history = [
    {"role": "system", "content": prompt},
]

def model_response(userResponse):
    global polling
    global conversation_history
    if userResponse == "Exit":
        print("exiting application")
        polling = False
        return
    userResponse += "\n Use the following data to assist in your reponse:"
    query_result = query(userResponse)
    for match in query_result["matches"]:
        var = match["metadata"]["original_data"]
        userResponse += f"{var}"

    conversation_history.append({"role": "user", "content": userResponse})
    response = client.chat.completions.create(
        model="gpt-4o-mini"  ,
        messages=conversation_history
    )
    conversation_history.append({"role":"assistant", "content":response.choices[0].message.content})
    print(f"{response.choices[0].message.content}\n")

def query(query_text):
    query_response = client.embeddings.create(input=[query_text], model="text-embedding-ada-002").data[0].embedding

    # Perform similarity search in Pinecone
    query_result = index.query(
        vector=query_response,
        top_k=2,  # Retrieve the 2 closest matches
        include_metadata=True  # Optional: include metadata in the response if stored
    )
    return query_result

def gpt_setup():
    global polling
      
    
    global conversation_history
    # Let the assistant initiate the conversation
    initial_response = client.chat.completions.create(
        model="gpt-4o-mini"  ,
        messages=conversation_history
    )
    #opening pitch
    conversation_history.append({"role": "assistant", "content": initial_response.choices[0].message.content})
    print(f"{initial_response.choices[0].message.content}\n")
    recorder = AudioToTextRecorder()
    while polling:
       recorder.text(model_response)
# Entry point
if __name__ == '__main__':
    gpt_setup()
