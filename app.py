import streamlit as st
from audio import SalesAssistant
from RealtimeSTT import AudioToTextRecorder

# Streamlit App Configuration


import streamlit as st
from audio import SalesAssistant

def main():
    # Set up the page configuration for better aesthetics
    st.set_page_config(page_title="Sales Assistant Interaction", page_icon="🤖", layout="centered")

    # Display a welcoming message with a heading and description
    st.title("🎙️ Welcome to the Sales Assistant!")
    st.markdown("""
    _Speak into the microphone and interact with the Sales Assistant. The assistant will respond to your queries._
    """)

    # Initialize the Sales Assistant
    agent1 = SalesAssistant()
    

    # Display the opening pitch with some emphasis
    st.markdown("### 📢 Opening Pitch")
    opening_pitch = agent1.opening_pitch()
    st.write(f"**Assistant:** {opening_pitch}")

    # Add some spacing for visual clarity
    st.markdown("---")

    # Simulate the conversation loop with a more structured display
    st.markdown("### 🎤 Start Speaking")
    should_end = False
    i = 0
    while not should_end:
        st.write(f"**Step {i+1}:**")
        st.write("🎙️ **User:** Speak into the microphone...")
        
        # Simulate listening and transcribing
        transcribed_text = agent1.listen()
        if transcribed_text:
            st.write(f"**User:** {transcribed_text}")
        
            # Simulate assistant response
            model_reply, should_end = agent1.model_response(transcribed_text)
            st.write(f"**SAPP:** {model_reply}")
        
        # Add some space between each interaction
        st.markdown("---")
        i += 1

    # End the interaction with a message
    st.markdown("### 🔚 Conversation Ended")
    st.write("Thank you for your time! The conversation has concluded.")
    st.write("Feel free to restart the interaction by refreshing the page.")

# Run the app
if __name__ == "__main__":
    main()


x = """import streamlit as st
from audio import SalesAssistant
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    agent1 = SalesAssistant()
    opening_pitch = agent1.opening_pitch()
    print(opening_pitch)
    #create user box
    while True:
        #create user box
        transcribedText = agent1.listen()
        print(transcribedText)
        #fill in user box with transcribed Text

        agentReply = agent1.model_response(transcribedText)
        print(agentReply)
        #create agent box
        #fill agent box
"""
