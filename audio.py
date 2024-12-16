from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import os
from RealtimeSTT import AudioToTextRecorder
import pyautogui
class SalesAssistant:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] =os.environ.get('OPENAI_KEY')
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_KEY'))
        self.index = self.pc.Index("sapp2")
        self.client = OpenAI()
        self.polling = True
        self.recorder = AudioToTextRecorder()
        self.prompt="You are a sales rep for a particular Solar panel installation agency. You will initiate a sales conversation with a customer. Speak like a sales agent actually would on the phone. Only use company information given to you alongside the question by the user. Be persuasive but simple in your responses. The goal is to be personalable and helpful to the client in order to sell the companies product"
        self.conversation_history = [
            {"role": "system", "content": self.prompt},
        ]
        self.conversation_history_end = [
            {"role": "system", "content": "In the following message, the user will provide an excerpt from a sales call between a sales agent and a user. I want you to determine if the sentiment of the sales message indicates the call is over, and the sales agent should subsequently hang up the call. If this is the case, reply only as True . If this is not the case, and it is clear that the conversation should continue, respond only as False"},
        ]
    def opening_pitch(self):
        initial_response = self.client.chat.completions.create(
            model="gpt-4o-mini"  ,
            messages=self.conversation_history
        )
        #opening pitch
        self.conversation_history.append({"role": "assistant", "content": initial_response.choices[0].message.content})
        return initial_response.choices[0].message.content
    def model_response(self, userResponse):
        if userResponse == "Exit":
            print("exiting application")
            self.polling = False
            return
        userResponse += "\n Use the following data to assist in your reponse:"
        query_result = self.query(userResponse)
        for match in query_result["matches"]:
            var = match["metadata"]["original_data"]
            userResponse += f"{var}"

        self.conversation_history.append({"role": "user", "content": userResponse})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini"  ,
            messages=self.conversation_history
        )
        self.conversation_history.append({"role":"assistant", "content":response.choices[0].message.content})

        self.conversation_history_end.append({"role": "user", "content":response.choices[0].message.content})
        response_end = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history_end
        )
        self.conversation_history_end.pop()
        print("\n\n")
        print(response_end.choices[0].message.content)
        if response_end.choices[0].message.content == "False":
            print("yooooo")
            should_end = False
        else:
            should_end = True
        print(f"\n\n\n{should_end}\n\n\n")
        #print(f"{response.choices[0].message.content}\n")
        return (response.choices[0].message.content, should_end)
    
    def listen(self):
        transText  = self.recorder.text()
        return transText
    def query(self, query_text):
        query_response = self.client.embeddings.create(input=[query_text], model="text-embedding-ada-002").data[0].embedding

        # Perform similarity search in Pinecone
        query_result = self.index.query(
            vector=query_response,
            top_k=2,  # Retrieve the 2 closest matches
            include_metadata=True  # Optional: include metadata in the response if stored
        )
        return query_result
