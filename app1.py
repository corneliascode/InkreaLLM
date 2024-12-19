import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from langchain.schema import HumanMessage, AIMessage
from graphs import graph
import time


st.set_page_config(page_title="InkreaLLM", page_icon="📝",layout="wide")

if 'first_page' not in st.session_state:
    st.session_state.first_page = True

def get_message_key(last_event):
    message_key = [key for key in last_event.keys() if "messages" in key][0]
    return message_key

def graph_messages_to_streamlit(last_event):
    message_key = get_message_key(last_event)
    messages = last_event[message_key]
    result = []
    for message in messages:
        if hasattr(message, "content"):
            if message.type == "human":
                result.append({"role": "user", "content": message.content})
            else:
                result.append({"role": "assistant", "content": message.content})
    return result

def generate_dynamic_markdown(data):
    markdown = "### Character Information\n\n"

    # Find the key containing the list of characters dynamically
    characters_key = next((key for key, value in data.items() if isinstance(value, list)), None)
    if not characters_key:
        return "No character data found."

    # Process each character in the dynamically found list
    for char in data[characters_key]:
        # Dynamically find the 'character_attributes' key
        attributes_key = next((key for key in char if isinstance(char[key], dict)), None)
        if not attributes_key:
            continue  # Skip this character if 'character_attributes' is not found

        attributes = char[attributes_key]
        
        # Dynamically extract the character's name (if it exists)
        name = attributes.get("name", "Unknown")
        markdown += f"## {name}\n"
        
        # Iterate through the character's attributes
        for key, value in attributes.items():
            # Handle nested dictionaries (e.g., 'appearance') and lists (e.g., 'skills')
            if isinstance(value, dict):
                markdown += f"- **{key.replace('_', ' ').capitalize()}**:\n"
                for sub_key, sub_value in value.items():
                    markdown += f"  - {sub_key.replace('_', ' ').capitalize()}: {sub_value}\n"
            elif isinstance(value, list):
                markdown += f"- **{key.replace('_', ' ').capitalize()}**: {', '.join(value)}\n"
            else:
                markdown += f"- **{key.replace('_', ' ').capitalize()}**: {value}\n"
        markdown += "\n"  # Add spacing between characters

    return markdown

def generate_dynamic_markdown_story(data):
    """
    Generate a Markdown representation of writing units from a structured dictionary.
    
    Args:
        data (dict): A dictionary containing unit information
    
    Returns:
        str: A Markdown-formatted string describing the writing's units
    """

    markdown = "### Character Information\n\n"
    # Find the key containing the list of units dynamically
    units_key = next((key for key, value in data.items() if isinstance(value, list)), None)
    if not units_key:
        return "No unit data found."

    # Process each chapter in the dynamically found list
    for index, unit in enumerate(data[units_key], 1):
        # Extract chapter details
        chapter_name = unit.get('unit_name', f'Chapter {index}')
        chapter_length = unit.get('unit_length', 'Unknown length')
        chapter_summary = unit.get('unit_summary', 'No summary available')

        # Create Markdown for each chapter
        markdown += f"## {chapter_name}\n\n"
        markdown += f"**Length:** {chapter_length}\n\n"
        markdown += f"**Summary:** {chapter_summary}\n\n"

    return markdown

def first_page():
    # Create a placeholder for the GIF
    gif_placeholder = st.empty()
    
    # Display the loading GIF
    gif_placeholder.image("https://cdn.dribbble.com/users/154752/screenshots/1244719/book.gif", width=800)
    
    # Loading stages with spinners
    with st.spinner('Preparing pencil and paper...'):
        time.sleep(2)  # Simulate the first stage of loading
    
    with st.spinner('Sharpening the pencil...'):
        time.sleep(2)  # Simulate sharpening the pencil
    
    with st.spinner('Getting the paper ready...'):
        time.sleep(2)  # Simulate preparing the paper
    
    with st.spinner('Setting up your workspace...'):
        time.sleep(2)  # Simulate final preparation steps
    
    # Clear the GIF placeholder
    gif_placeholder.empty()
    
    # Set a flag in session state to indicate loading is done
    st.session_state.loading_complete = True

    st.session_state.first_page = False
   


def main(): 
    if st.session_state.first_page:
        first_page()

    # Initialize session state variables if they don't exist
    if 'events' not in st.session_state:
        config = {"configurable": {"thread_id": "13", "recursion_limit": 1000}}
        st.session_state.events = list(graph.stream(
            {"messages": [], "story_info": []}, 
            config, 
            stream_mode="values",
            subgraphs=True
        ))


     # Ensure last_event is defined by getting the last event from session state
    last_event = st.session_state.events[-1] if st.session_state.events else None


    # page_by_img ="""
    # <style>
    # [data-testid="stAppViewContainer"] {
    # background-image: url("https://img.freepik.com/premium-photo/stack-books-black-background_1247484-16920.jpg?semt=ais_hybrid");
    # background-size: cover;
    # background-attachment: local;
    # }

    # [data-testid="stHeader"] {
    # background-color: rgba(0,0,0,0);
    # }
    # </style>
    # """
    # st.markdown(page_by_img, unsafe_allow_html=True)

    
    col1, col2 = st.columns(2)
    

    with col1:
        # Display story info in a container
        st.header("Story info")
        story_container = st.container(height=150)
        with story_container:
            if 'story_info' in last_event[1]:
                story_info = last_event[1].get("story_info")
                st.markdown(story_info[0].content if len(story_info) > 0 else "")
            
                       
        st.header("Let's write something")
        
        # Display chat history in a container
        history_container = st.container(height=320)
        
        # Get the last event and messages
        last_event = st.session_state.events[-1]
        messages_list = graph_messages_to_streamlit(last_event[1])
        
        with history_container:
            for message in messages_list:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        user_message = st.text_area("Type your message:", height=70)
        if st.button("Send"):
            # Update the state with the new human message
            config = {"configurable": {"thread_id": "13", "recursion_limit": 1000}}
            graph_state = graph.get_state(config, subgraphs=True)
            subgraph_config = graph_state.tasks[0].state.config
            subgraph_config["configurable"]['recursion_limit'] = 1000
            message_key = get_message_key(last_event[1])
            
            subgraph_name = subgraph_config.get("configurable").get("checkpoint_ns").split(':')[0]
            if subgraph_name in ["story_writer", "character_supervisor", "story_structure_creator"]:
                message_to_send = [AIMessage(content=messages_list[-1]["content"]), HumanMessage(content=user_message)]
            else:
                message_to_send = [HumanMessage(content=user_message)]
            # change value of need_answer based on the subgraph
            if subgraph_name == "story_structure_creator":
                story_structure_has_answer = True
            else:
                story_structure_has_answer = False
            if subgraph_name == "character_supervisor":
                characters_strcture_has_answer = True
            else:
                characters_strcture_has_answer = False

            graph.update_state(
                config=subgraph_config, 
                values={
                    message_key: message_to_send, 
                    "agent_007_need_answer": False,
                    "additional_info_gatherer_need_answer": False,
                    "next_paragraph_writer_has_answer": True,
                    "paragraph_rewriter_has_answer": True,
                    "characters_strcture_has_answer": characters_strcture_has_answer,
                    "story_structure_has_answer": story_structure_has_answer
                }
            )
            
            # Get new events after update
            st.session_state.events = list(graph.stream(None, config, 
                                                        stream_mode="values", 
                                                        subgraphs=True))
            
            # Rerun the app to refresh the display
            st.rerun()

    with col2:
        ch_i_height = 150
        st_c_height = 450
        if "characters_changed" in last_event[1]:
            if last_event[1].get("characters_changed") == True:
                ch_i_height = 600
                st_c_height = 0

        # Display character info in a container
        st.header("Character info")
        character_container = st.container(height=ch_i_height)
        with character_container:
            if ('character_structured_info' in last_event[1]) or ("characters_changed" in last_event[1]):
                message_key = get_message_key(last_event[1])
                try:
                    to_present = eval(last_event[1].get(message_key)[-1].content)
                except:
                    to_present = last_event[1].get("character_structured_info")
                st.markdown(generate_dynamic_markdown(to_present))
        # Display story content in a container
        if len(last_event[1].get("story_content", [])) > 0:
            st.header("Story content")
        else:
            st.header("Story structure")
        story_content_container = st.container(height=st_c_height)
        with story_content_container:
            if len(last_event[1].get("story_content", [])) > 0:
                if 'story_content' in last_event[1]:
                    story_content = last_event[1].get("story_content")
                    try:
                        story_content = [element.content for element in story_content] if len(story_content) > 0 else ""
                        story_content = "\n\n".join(story_content) if isinstance(story_content, list) else story_content
                        st.markdown(story_content)
                    except Exception as e:
                        pass
            else:
                if last_event[0][0].split(':')[0] == "story_structure_creator":
                    message_key = get_message_key(last_event[1])
                    try:
                        to_present = eval(last_event[1].get(message_key)[-1].content)
                    except:
                        to_present = ""
                    st.markdown(generate_dynamic_markdown_story(to_present))


if __name__ == "__main__":
    main()