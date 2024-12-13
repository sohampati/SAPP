from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index("sapp2")

client = OpenAI()

data = {
  "company_name": "SunWave Energy Solutions",
  "products_0.name": "SunWave MonoPlus Panels",
  "products_0.type": "Monocrystalline",
  "products_0.efficiency": 22,
  "products_0.wattage": 400,
  "products_0.warranty_years": 25,
  "products_0.warranty_performance": 80,
  "products_0.price_per_watt": 2.8,
  "products_0.dimensions": "77 x 39 inches",
  "products_0.weight": 44,
  "products_0.degradation_rate": 0.5,
  "products_1.name": "EcoWave Poly Panels",
  "products_1.type": "Polycrystalline",
  "products_1.efficiency": 18,
  "products_1.wattage": 330,
  "products_1.warranty_years": 20,
  "products_1.price_per_watt": 2.3,
  "products_1.dimensions": "75 x 39 inches",
  "products_1.weight": 46,
  "products_2.name": "FlexiWave Thin-Film Panels",
  "products_2.type": "Thin-Film",
  "products_2.efficiency": 12,
  "products_2.wattage": 100,
  "products_2.warranty_years": 15,
  "products_2.price_per_watt": 1.9,
  "products_2.dimensions": "60 x 30 inches",
  "products_2.weight": 15,
  "services_0": "System customization",
  "services_1": "Annual maintenance",
  "services_2": "24/7 system monitoring",
  "services_3": "System upgrades",
  "warranties.inverter_years": 12,
  "warranties.inverter_extendable_years": 20,
  "warranties.installation_workmanship_years": 10,
  "financing_options_0": "Solar Lease",
  "financing_options_1": "Solar Loan",
  "financing_options_2": "Power Purchase Agreement (PPA)",
  "incentives.federal_tax_credit_percentage": 30,
  "incentives.california_state_rebate": 1500,
  "installation_timeline.site_inspection_days": 2,
  "installation_timeline.design_and_permitting_weeks": 3,
  "installation_timeline.installation_days": 3,
  "installation_timeline.testing_and_activation_weeks": 1,
  "installation_timeline.total_project_weeks": 8,
  "customer_benefits.average_bill_reduction_percentage": 80,
  "customer_benefits.lifetime_savings_estimate": 25000,
  "customer_benefits.co2_offset_pounds": 100000,
  "customer_benefits.equivalent_trees_planted": 1500,
  "system_compatibility.inverters_0": "SolarEdge",
  "system_compatibility.inverters_1": "Enphase",
  "system_compatibility.batteries_0": "SunWave Storage Pro",
  "system_compatibility.smart_home_integrations_0": "Smart thermostats",
  "system_compatibility.smart_home_integrations_1": "EV chargers",
  "system_compatibility.smart_home_integrations_2": "Home energy management systems",
  "panel_durability.wind_resistance_mph": 140,
  "panel_durability.snow_load_rating_pa": 5400,
  "sales_process_0": "Initial contact",
  "sales_process_1": "Free consultation",
  "sales_process_2": "Energy usage analysis",
  "sales_process_3": "Roof assessment",
  "sales_process_4": "Custom proposal",
  "quote_turnaround_days": 3
}

data = [f"{key}:{value}" for key, value in data.items()]

def upsert():
    index.delete(delete_all=True)
    embeddings = []
    for text in data:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
        embeddings.append({"id": text, "values": response, "metadata": {"original_data": text}})

    # Upsert embeddings to Pinecone with metadata
    index.upsert(vectors=embeddings)
    print("Upsertion complete")
    response = index.describe_index_stats()
    print(response)

#upsert()
# Query for the closest matches
def query(query_text):
    query_response = client.embeddings.create(input=[query_text], model="text-embedding-ada-002").data[0].embedding

    # Perform similarity search in Pinecone
    query_result = index.query(
        vector=query_response,
        top_k=2,  # Retrieve the 2 closest matches
        include_metadata=True  # Optional: include metadata in the response if stored
    )
    return query_result



model = "gpt-4o-mini"
prompt="You are a sales rep for a particular Solar panel installation agency. You will initiate a sales conversation with a customer. Speak like a sales agent actually would on the phone. Only use company information given to you alongside the question by the user. Be persuasive but simple in your responses. The goal is to be personalable and helpful to the client in order to sell the companies product"
conversation_history = [
    {"role": "system", "content": prompt},
]
# Let the assistant initiate the conversation
initial_response = client.chat.completions.create(
    model=model,
    messages=conversation_history
)
#opening pitch
conversation_history.append({"role": "assistant", "content": initial_response.choices[0].message.content})
print(f"{initial_response.choices[0].message.content}\n")
userResponse = input()
while userResponse != "exit":
    userResponse += "\n Use the following data to assist in your reponse:"
    query_result = query(userResponse)
    for match in query_result["matches"]:
        userResponse += f"\n{match["metadata"]["original_data"]}"

    conversation_history.append({"role": "user", "content": userResponse})
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history
    )
    conversation_history.append({"role":"assistant", "content":response.choices[0].message.content})
    print(f"{response.choices[0].message.content}\n")
    userResponse = input()