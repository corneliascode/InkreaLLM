import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langgraph.errors import NodeInterrupt
import time

load_dotenv(override=True)

memory = MemorySaver()

# llm_objects
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

openai_strict_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GEMINI_API_KEY")
)

# SUBGRAPHS

# INFO GATHERER SUBGRAPH
class InfoGathererSubgraphState(TypedDict):
    temp_messages: Annotated[list, add_messages]
    story_info: Annotated[list, add_messages]
    agent_007_need_answer: bool
    additional_info_gatherer_need_answer: bool

def agent_007(state: InfoGathererSubgraphState):
    if state.get("agent_007_need_answer") == True:
        raise NodeInterrupt(
            f"Question interrupt.")
    # Count questions by checking message content
    #(messages must be type AI & Message content is not "FINISH")
    question_count = sum(1 for msg in state["temp_messages"] 
                        if hasattr(msg, 'type') and msg.type == 'ai' 
                        and msg.content != "FINISH")
    
    conversation_history = "\n".join([
        f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" 
        for msg in state["temp_messages"]
    ])

    story_info_extraction_template = PromptTemplate.from_template("""
    You are a very talented writer initiating a writing project.

    INITIAL STEP: Determine the Type of Writing
    First, you must identify the specific type of writing the user wants to create. Ask the user to clarify:
    - Are they interested in writing a novel, short story, screenplay, poem, non-fiction, or another form of writing?
    - Clarify the specific genre or category within that writing type

    Once the type of writing is established, proceed with the following strict instructions:
    0. Ask ONLY ONE question at a time.
    1. You have asked {question_count} questions so far.
    2. You must ask EXACTLY 5 QUESTIONS IN TOTAL! 
    3. The questions should be deeply tailored to the specific writing type identified.
    4. Take into consideration previous conversation when formulating the question.
    5. Previous conversation:
    {information}

    Genre-Specific Question Frameworks:
    - Fiction Novel: Character motivations, world-building, plot complexity
    - Short Story: Central conflict, narrative arc, thematic core
    - Poetry: Emotional landscape, structural intentions, inspirational sources
    - Screenplay: Character dynamics, narrative tension, visual storytelling

    If {question_count} IS EQUAL TO 5, respond ONLY with the word "FINISH".
    
    IMPORTANT: When responding with "FINISH", send ONLY the word "FINISH" without any other text.
""")

    chain = story_info_extraction_template | openai_llm
    response = chain.invoke({
        "information": conversation_history,
        "question_count": question_count
    })
    time.sleep(1)
    
    # Create AIMessage
    if response.content == "FINISH":
        return {
            "temp_messages": [AIMessage(content=response.content)],
            "agent_007_need_answer": False,
            "additional_info_gatherer_need_answer": False
        }

    return {
        "temp_messages": [AIMessage(content=response.content)],
        "agent_007_need_answer": True
    }

def additional_info_gatherer(state: InfoGathererSubgraphState):
    if state.get("additional_info_gatherer_need_answer") == True:
        raise NodeInterrupt(
            f"Additional info interrupt.")
    if state["temp_messages"][-1].content == "":
        return {"temp_messages": [AIMessage(content="FINISH")]}
    additional_info_template = PromptTemplate.from_template("""
    You are a very helpful assistant.

    Write ONLY in ENGLISH!

    Check the previous message: {previous_answer}
                                                            
    1. Prompt the user by asking: "Do you have any additional details or information you would like to provide for this writing project?"

    2. User Response Handling:
    a) If the user confirms they have NO additional information respond ONLY with the word: "FINISH"
    b) If the user indicates they DO have additional information ask to provide the specific additional details or information for the writing project                                                                                                            
    """)
    chain = additional_info_template | openai_llm
    response = chain.invoke({"previous_answer": state["temp_messages"][-1].content})
    time.sleep(1)
    if response.content == "FINISH":
        return {"temp_messages": [AIMessage(content=response.content)], "additional_info_gatherer_need_answer": True}
    return {"temp_messages": [AIMessage(content=response.content)], "additional_info_gatherer_need_answer": True}

def info_condenser(state: InfoGathererSubgraphState):
    story_info_condenser_template = PromptTemplate.from_template("""
    You are a very talented writer.
    Please extract from {information} in a list all the information provided by the user about the writing he want (asked questions and answers).
    Respond with a structure ["question: answer"] as plain text.                                                               
    """)
    chain = story_info_condenser_template | openai_llm
    response = chain.invoke({"information": state["temp_messages"]})
    return {"story_info": [response.content]}

def chatbot_router1(
    state: InfoGathererSubgraphState,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("temp_messages", []):    # Trying to get "temp_messages" from the state dictionary & Assigning the result to the messages variable
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if ai_message.content == "FINISH":
        return "additional_info"
    return "info_asker"

def chatbot_router2(
    state: InfoGathererSubgraphState,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("temp_messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if ai_message.content == "FINISH":
        return "info_condenser"
    return "additional_info"

inf_gatherer_subgraph_builder = StateGraph(InfoGathererSubgraphState)
inf_gatherer_subgraph_builder.add_node("info_asker", agent_007)
inf_gatherer_subgraph_builder.add_node("additional_info", additional_info_gatherer)
inf_gatherer_subgraph_builder.add_node("info_condenser", info_condenser)
inf_gatherer_subgraph_builder.add_edge(START, "info_asker")

# # After the "info_asker" node, chatbot_router1 decides the next step
# If more information is needed, it goes to "additional_info", 
# If not, it can loop back to "info_asker"
inf_gatherer_subgraph_builder.add_conditional_edges(
    "info_asker",
    chatbot_router1,
    {"additional_info": "additional_info", "info_asker": "info_asker"},
)

# # After the "additional_info" node, chatbot_router2 decides the next step
# If more information is needed, it can go back to "additional_info"
# If information is complete, it moves to the "info_condenser"
inf_gatherer_subgraph_builder.add_conditional_edges(
    "additional_info",
    chatbot_router2,
    {"additional_info": "additional_info", "info_condenser": "info_condenser"},
)
#Finalizes the state graph, creating a compiled workflow
inf_gatherer_subgraph = inf_gatherer_subgraph_builder.compile(checkpointer=memory)


# STORY WRITER SUBGRAPH

class StoryWriterSubgraphState(TypedDict):
    temp_messages: Annotated[list, add_messages]
    story_info: Annotated[list, add_messages]
    character_structured_info: dict
    story_structured_info: dict
    story_content: list
    actual_unit: list
    state_user_input: str
    paragraph_to_change: str
    unit_length: str
    next_paragraph_writer_has_answer: bool
    paragraph_rewriter_has_answer: bool
    characters_changed: bool
    
def next_paragraph_writer(state: StoryWriterSubgraphState):
    if state.get("next_paragraph_writer_has_answer") != True:
        story_content = state.get("story_content", [])
        unit_length = state.get("unit_length", [])
        no_previous_paragraphs = len(story_content) - 1
        
        next_paragraph_writer_template = PromptTemplate.from_template("""
        
        You are an advanced narrative writer tasked with continuing a story with precision and creative depth.
        Context:
        - Character Details: {character_structured_info}
        - Story Structure: {story_structured_info}
        - Previous Narrative: {story_content}
        - Current Progress: {no_previous_paragraphs} of {unit_length} paragraphs completed
        - Current Unit Description: unit description of the current unit from the Writing structure
        
        Writing Guidelines:
        1. Narrative Continuity
        - Maintain consistent style, tone, and narrative flow
        - Align with previous paragraphs and story structure
        - Seamlessly integrate current unit description

        2. Dialogue Formatting
        - Use em dash (—) for dialogue
        - Separate dialogue from narrative text
        - Include dialogue only when narratively necessary
        - Example:
        — Do you want something like this? asks Mike.
        — Then you learn, man! replies Tom.
        — Mike's encouragement sparked a glimmer of hope.

        3. Character Interaction
        - Ensure character actions and reactions consistently reflect:
           - Individual character traits
           - Current story context
           - Interpersonal dynamics
                                                                      
        Evaluation Criteria:
        - Narrative progression
        - Character authenticity
        - Structural alignment
        - Stylistic consistency

        Response Protocol. IMPORTANT!:
        - If <{unit_length}> is bigger than <{no_previous_paragraphs}>:
           - Write next paragraph
           - Maintain narrative coherence
        - If <{no_previous_paragraphs}> is equal or bigger than <{unit_length}>:
           - Respond ONLY with "FINISH", nothing else!

        """)
        chain = next_paragraph_writer_template | openai_llm
        response = chain.invoke({"story_content": story_content,
                                 "character_structured_info": str(state["character_structured_info"]),
                                 "story_structured_info": str(state["story_structured_info"]),
                                 "unit_length": unit_length, "no_previous_paragraphs": no_previous_paragraphs})
        state['temp_messages'].append(AIMessage(content=response.content))
        time.sleep(2)
        raise NodeInterrupt(
                f"Next paragraph writer interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = state.get("temp_messages")[-2]
    story_content = state.get("story_content", [])
    actual_unit = state.get("actual_unit", [])
    if (user_input == "") and (response.content != "FINISH"):
        story_content.append(response)
        actual_unit.append(response)
        return {"story_content": story_content,
                "actual_unit": actual_unit,
                "state_user_input": user_input, 
                "next_paragraph_writer_has_answer": False,
                "characters_changed": False}
    elif (user_input == "") and (response.content == "FINISH"):
        return {"temp_messages": [AIMessage(content="FINISH")], 
                "characters_changed": False}
    elif user_input == "FINISH":
        story_content.append(response)
        actual_unit.append(response)
        return {"temp_messages": [AIMessage(content="FINISH")], 
                "story_content": story_content,
                "actual_unit": actual_unit,
                "state_user_input": user_input,
                "characters_changed": False}
    return {"state_user_input": user_input, 
            "paragraph_to_change": response.content,
            "paragraph_rewriter_has_answer": False,
            "characters_changed": False}

def paragraph_rewriter(state: StoryWriterSubgraphState):
    if state.get("paragraph_rewriter_has_answer") != True:
        user_input = state.get("state_user_input", "")
        story_content = state.get("story_content", [])
        paragraph_to_change = state.get("paragraph_to_change", "")    
        paragraph_rewriter_template = PromptTemplate.from_template("""
        You are a very talented writer working on a writing.

        Context:
            - Characters Information: {character_structured_info}
            - Writing structure: {story_structured_info}
            - Previous Paragraphs: {story_content}
            - User Input: {user_input}
            - Paragraph to be Changed: {paragraph_to_change}

        Task:
            - Rewrite the paragraph to be changed based on the requirements from the user input.


        Response Requirements:
            - The paragraph should be rewritten based on the user input requirements.
            - For keeping the paragraph attributes, OTHER that those required to be changed, use the characters information, writing structure and previous paragraphs.
        """)
        chain = paragraph_rewriter_template | openai_llm
        response = chain.invoke({"story_content": story_content, 
                                 "user_input": user_input, "paragraph_to_change": paragraph_to_change,
                                 "character_structured_info": str(state["character_structured_info"]),
                                 "story_structured_info": str(state["story_structured_info"])})
        state['temp_messages'].append(AIMessage(content=response.content))
        time.sleep(2)
        raise NodeInterrupt(
                f"Paragraph rewriter interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = state.get("temp_messages")[-2]
    story_content = state.get("story_content", [])
    actual_unit = state.get("actual_unit", [])
    if (user_input == "") and (response.content != "FINISH"):
        story_content.append(response)
        actual_unit.append(response)
        return {"story_content": story_content,
                "actual_unit": actual_unit,
                "state_user_input": user_input, 
                "next_paragraph_writer_has_answer": False,
                "characters_changed": False}
    elif (user_input == "") and (response.content == "FINISH"):
        return {"temp_messages": [AIMessage(content="FINISH")], 
                "characters_changed": False}
    elif (user_input == "FINISH"):
        story_content.append(response)
        actual_unit.append(response)
        return {"temp_messages": [AIMessage(content="FINISH")], 
                "story_content": story_content,
                "actual_unit": actual_unit,
                "state_user_input": user_input,
                "characters_changed": False}
    return {"state_user_input": user_input,
            "paragraph_to_change": response.content,
            "paragraph_rewriter_has_answer": False,
            "characters_changed": False}

def chatbot_router3(
    state: StoryWriterSubgraphState,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("temp_messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    user_input = state.get("state_user_input", "")
    if (ai_message.content != "FINISH") and (user_input == ""):
        return "next_paragraph_writer"
    elif (ai_message.content != "FINISH") and (user_input != ""):
        return "paragraph_rewriter"
    return "END"
    

story_writer_subgraph_builder = StateGraph(StoryWriterSubgraphState)
story_writer_subgraph_builder.add_node("next_paragraph_writer", next_paragraph_writer)
story_writer_subgraph_builder.add_node("paragraph_rewriter", paragraph_rewriter)

story_writer_subgraph_builder.add_edge(START, "next_paragraph_writer")

story_writer_subgraph_builder.add_conditional_edges(
    "next_paragraph_writer",
    chatbot_router3,
    {"next_paragraph_writer": "next_paragraph_writer", "paragraph_rewriter": "paragraph_rewriter", "END": END},
)


story_writer_subgraph_builder.add_conditional_edges(
    "paragraph_rewriter",
    chatbot_router3,
    {"next_paragraph_writer": "next_paragraph_writer", "paragraph_rewriter": "paragraph_rewriter", "END": END},
)
#Finalizes the state graph, creating a compiled workflow
story_writer_subgraph = story_writer_subgraph_builder.compile(checkpointer=memory)



# CHARACTER SUPERVISOR SUBGRAPH

class CharacterSupervisorSubgraphState(TypedDict):
    temp_messages: Annotated[list, add_messages]
    story_info: Annotated[list, add_messages]
    story_content: Annotated[list, add_messages]
    characters_info: Annotated[list, add_messages]
    character_structured_info: dict
    character_description_structure: dict
    characters_changed: bool
    characters_strcture_has_answer: bool
    state_user_input: str

class CharacterStructure(TypedDict):
    character_attributes: Annotated[dict, "Character attributes or traits"]

class CharactersList(TypedDict):
        character : Annotated[dict, "List of characters"]

class CharactersStructuredInfo(TypedDict):
    characters : Annotated[dict, "List of characters with structured information"]

def character_structure_creator(state: CharacterSupervisorSubgraphState):
    structured_openai_llm = openai_llm.with_structured_output(CharacterStructure, method="json_mode")
    character_structure_creator_template = PromptTemplate.from_template("""
    You are a very talented writer.
    Taking into consideration the information provided by the user about the writing attributes he want, from <{story_info}>, 
    please create a set of atributes or traits that you would need in order to define any character that could appear in the writing.
    The character could be human or not. The attributes should be general and not specific to any character.
    Do not use relations or history, since the list will be populated before starting to write the writing.
    Respond in JSON.
    """)
    chain = character_structure_creator_template | structured_openai_llm
    response = chain.invoke({"story_info": state["story_info"]})
    return {"temp_messages": [AIMessage(content=str(response))], "character_description_structure": [str(response)]}


def caracter_extractor(state: CharacterSupervisorSubgraphState):
    structured_openai_llm = openai_llm.with_structured_output(CharactersList, method="json_mode")
    character_extractor_template = PromptTemplate.from_template("""
    You are a very talented writer.
    Taking into consideration the information provided by the user about the writing, from <{story_info}>, 
    please extract from the information provided by user the characters that appear in the writing.
    For each character provide a name (if there is a name in the information provided by the user - use it, otherwise - generate one),
    and a description of the character if here is one in the information provided by the user, otherwise - leave blank the description.
    If there is no characters - respond with an empty list.
    Respond in JSON.
    """)
    chain = character_extractor_template | structured_openai_llm
    response = chain.invoke({"story_info": state["story_info"]})
    return {"temp_messages": [AIMessage(content=str(response))], 
            "characters_info": [str(response)], 
            'characters_changed': True}

def character_description_creator(state: CharacterSupervisorSubgraphState):
    if state.get("characters_strcture_has_answer") != True:
        structured_openai_llm = openai_llm.with_structured_output(CharactersStructuredInfo, method="json_mode")
        character_description_creator_template = PromptTemplate.from_template("""
        You are a very talented writer.
        Taking into consideration the information provided by the user about the writing, from <{story_info}> and the list of characters from <{characters_info}>
        please describe each character following the structure <{character_info_structure}> for each character in the provided list of characters.
        In the descriptions, include only values, without types.
        If there are no characters in the list of characters - return an empty list.
        The attributes and traits for each character should be taken from the list of characters when the needed information is there, 
        otherwise, infer them based on the information provided by the user about the writing.
        Respond in JSON.
        """)
        chain = character_description_creator_template | structured_openai_llm
        response = chain.invoke({"story_info": state["story_info"], 
                                "characters_info": state["characters_info"],
                                 "character_info_structure": state["character_description_structure"]}
                                 )
        state['temp_messages'].append(AIMessage(content=str(response)))
        time.sleep(2)
        raise NodeInterrupt(
                f"Character structure writer interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = eval(state.get("temp_messages")[-2].content)
    if (user_input == "") or (user_input == "FINISH"):
        return {"temp_messages": [AIMessage(content=str(response))],
                "character_structured_info": response, 
                "state_user_input": user_input, 
                "characters_strcture_has_answer": False,
                "characters_changed": False}
    return {"temp_messages": [AIMessage(content=str(response))],
            "state_user_input": user_input,
            "character_structured_info": response,
            "characters_strcture_has_answer": False,
            "characters_changed": True}

def character_description_recreator(state: CharacterSupervisorSubgraphState):
    if state.get("characters_strcture_has_answer") != True:
        user_input = state.get("state_user_input", "")
        structured_openai_llm = openai_llm.with_structured_output(CharactersStructuredInfo, method="json_mode")
        character_description_recreator_template = PromptTemplate.from_template("""
        You are a very talented writer working on a writing.

            Context:
                - Writing Information: {story_info}
                - Characters Information: {character_structured_info}
                - User Input: {user_input}

            Task:
                - Rewrite the Characters Information based on the requirements from the user input. Keep the exact same structure.


            Response Requirements:
                - The Characters Information should be rewritten based on the user input requirements.
                - For keeping the characters information structure, OTHER that those required to be changed, use the writing information and characters information.
            Respond in JSON.
                """)
        chain = character_description_recreator_template | structured_openai_llm
        response = chain.invoke({"story_info": state["story_info"], 
                                "character_structured_info": state["character_structured_info"],
                                 "user_input": user_input}
                                 )
        state['temp_messages'].append(AIMessage(content=str(response)))
        time.sleep(2)
        raise NodeInterrupt(
                f"Character strcture writer interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = eval(state.get("temp_messages")[-2].content)
    if (user_input == "") or (user_input == "FINISH"):
        return {"temp_messages": [AIMessage(content=str(response))],
                "character_structured_info": response, 
                "state_user_input": user_input, 
                "characters_strcture_has_answer": False,
                "characters_changed": False}
    return {"temp_messages": [AIMessage(content=str(response))],
            "state_user_input": user_input,
            "character_structured_info": response,
            "characters_strcture_has_answer": False,
            "characters_changed": True}



def chatbot_router4(
    state: CharacterSupervisorSubgraphState,
):
    user_input = state.get("state_user_input", "")
    if (user_input == "") or (user_input == "FINISH"):
        return "END"
    else:
        return "character_description_recreator"
    

character_supervisor_subgraph_builder = StateGraph(CharacterSupervisorSubgraphState)
character_supervisor_subgraph_builder.add_node("character_structure_creator", character_structure_creator)
character_supervisor_subgraph_builder.add_node("caracter_extractor", caracter_extractor)
character_supervisor_subgraph_builder.add_node("character_description_creator", character_description_creator)
character_supervisor_subgraph_builder.add_node("character_description_recreator", character_description_recreator)

character_supervisor_subgraph_builder.add_edge(START, "character_structure_creator")
character_supervisor_subgraph_builder.add_edge("character_structure_creator", "caracter_extractor")
character_supervisor_subgraph_builder.add_edge("caracter_extractor", "character_description_creator")


character_supervisor_subgraph_builder.add_conditional_edges(
    "character_description_creator",
    chatbot_router4,
    {"character_description_recreator": "character_description_recreator", "END": END},
)

character_supervisor_subgraph_builder.add_conditional_edges(
    "character_description_recreator",
    chatbot_router4,
    {"character_description_recreator": "character_description_recreator", "END": END},
)

#Finalizes the state graph, creating a compiled workflow
character_supervisor_subgraph= character_supervisor_subgraph_builder.compile(checkpointer=memory)

# STORY STRUCTURE CREATOR SUBGRAPH

class StoryStructureCreatorState(TypedDict):
    temp_messages: Annotated[list, add_messages]
    story_info: Annotated[list, add_messages]
    character_structured_info: dict
    story_structured_info: dict
    story_structure_has_answer: bool
    state_user_input: str


class StoryStructure(TypedDict):
    story_structure : Annotated[dict, "Writing Structure"]

def story_structure_creator(state: StoryStructureCreatorState):
    if state.get("story_structure_has_answer") != True:
        structured_openai_llm = openai_llm.with_structured_output(StoryStructure, method="json_mode")
        story_structure_creator_template = PromptTemplate.from_template("""
        You are a very talented writer.
        Taking into consideration the information provided by the user about the writing, from <{story_info}> and the characters information from <{character_structured_info}>
        please create a structure for the writing. The structure should contain the units of the writing, the type of units depending on the type of writing.
        For each unit provide a title/name.
        The structure should be in accordance with the type of the writing and the information about it, 
        the titles/names and the units length should be in accordance with the writing type, characters information and the writing information.
        If the writing type cannot be inferred from the information provided by the user, use a generic structure.
        Respond in JSON. 
       
           Guideline examples:
            1. Novels: Chapters
            2. Short Stories: Sections
            3. Poems: Stanzas
            4. Screenplays: Scenes in 3-act structure

            Respond in JSON:
            {{
                "units": [
                    {{
                        "unit_type": "Chapter/Section/Stanza/Scene/etc.",
                        "unit_name": "Unique Unit Name",
                        "unit_length": "in paragraphs",
                        "unit_summary": "short summary about what could be written in the unit"                
                    }}
                ]
            }}
                """)
        
        chain = story_structure_creator_template | structured_openai_llm
        response = chain.invoke({"story_info": state["story_info"], 
                                 "character_structured_info": state["character_structured_info"]}
                                 )
        state['temp_messages'].append(AIMessage(content=str(response)))
        time.sleep(2)
        raise NodeInterrupt(
                f"Story structure writer interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = eval(state.get("temp_messages")[-2].content)
    if (user_input == "") or (user_input == "FINISH"):
        return {"temp_messages": [AIMessage(content=str(response))],
                "story_structured_info": response, 
                "state_user_input": user_input, 
                "story_structure_has_answer": False,
                "story_structure_changed": False}
    return {"temp_messages": [AIMessage(content=str(response))],
            "state_user_input": user_input,
            "story_structured_info": response,
            "story_structure_has_answer": False,
            "story_structure_changed": True}

def story_structure_recreator(state: StoryStructureCreatorState):
    if state.get("story_structure_has_answer") != True:
        user_input = state.get("state_user_input", "")
        structured_openai_llm = openai_llm.with_structured_output(StoryStructure, method="json_mode")
        story_structure_recreator_template = PromptTemplate.from_template("""
        You are a very talented writer working on a writing.

            Context:
                - Writing Information: {story_info}
                - Characters Information: {character_structured_info}
                - Story Structure: {story_structured_info}
                - User Input: {user_input}

            Task:
                - Rewrite the Story Structure based on the requirements from the user input. Keep the exact same structure.


            Response Requirements:
                - The Story Structure should be rewritten based on the user input requirements.
                - For keeping the Story Structure, OTHER that those required to be changed, use the writing information and characters information.
            Respond in JSON.
                """)
        chain = story_structure_recreator_template | structured_openai_llm
        response = chain.invoke({"story_info": state["story_info"], 
                                "character_structured_info": state["character_structured_info"],
                                "story_structured_info": state["story_structured_info"],
                                 "user_input": user_input}
                                 )
        state['temp_messages'].append(AIMessage(content=str(response)))
        time.sleep(2)
        raise NodeInterrupt(
                f"Story structure rewriter interrupt.")
    user_input = state.get("temp_messages")[-1].content
    response = eval(state.get("temp_messages")[-2].content)
    if (user_input == "") or (user_input == "FINISH"):
        return {"temp_messages": [AIMessage(content=str(response))],
                "story_structured_info": response, 
                "state_user_input": user_input, 
                "story_structure_has_answer": False,
                "story_structure_changed": False}
    return {"temp_messages": [AIMessage(content=str(response))],
            "state_user_input": user_input,
            "story_structured_info": response,
            "story_structure_has_answer": False,
            "story_structure_changed": True}

def chatbot_router5(
    state: StoryStructureCreatorState,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    user_input = state.get("state_user_input", "")
    if (user_input == "") or (user_input == "FINISH"):
        return "END"
    else:
        return "story_structure_recreator"
    
story_structure_creator_subgraph_builder = StateGraph(StoryStructureCreatorState)
story_structure_creator_subgraph_builder.add_node("story_structure_creator", story_structure_creator)
story_structure_creator_subgraph_builder.add_node("story_structure_recreator", story_structure_recreator)
story_structure_creator_subgraph_builder.add_edge(START, "story_structure_creator")


story_structure_creator_subgraph_builder.add_conditional_edges(
    "story_structure_creator",
    chatbot_router5,
    {"story_structure_recreator": "story_structure_recreator", "END": END},
)


story_structure_creator_subgraph_builder.add_conditional_edges(
    "story_structure_recreator",
    chatbot_router5,
    {"story_structure_recreator": "story_structure_recreator", "END": END},
)

story_structure_creator_subgraph = story_structure_creator_subgraph_builder.compile(checkpointer=memory)

# MAIN GRAPH

class MainGraphState(TypedDict):
    messages: Annotated[list, add_messages]
    story_info: Annotated[list, add_messages]
    character_structured_info: dict
    story_structured_info: dict
    story_content: list
    full_story: Annotated[list, add_messages]
    actual_unit: list
    unit_length: str
    characters_changed: bool



@tool
def save_text_to_file(text: str, filename: str) -> None:
    """Save a text to a file

    Args:
        text (str): text to save
        filename (str): the name of the file to save the text to, including the extension .txt
    """
    with open(filename, "w", encoding="utf-8") as output:
        output.write(text)

tools = [save_text_to_file]

def story_saver(state: MainGraphState):
    human_messages = [msg.content for msg in state.get("full_story", [])]
    text_to_save = "\n".join(human_messages)
    story_saver_template = PromptTemplate.from_template("""
    You are a very talented assistant.
    Please give me a name for the file where I will save the writing with the structure <{story_structured_info}>.
    The name should also be the title of the writing. Add the extension .txt at the end of the name.
    Respond in plain text, ONLY the name.                                                       
    """)
    chain = story_saver_template | gemini_llm
    response = chain.invoke({"story_structured_info": state["story_structured_info"]})
    tools[0].run(tool_input={
         "text": text_to_save, 
         "filename": str(response.content).strip()
     })
    return {"messages": AIMessage(content="The writing was saved successfully!")}

def structure_supervisor(state: MainGraphState):
    actual_unit = state.get("actual_unit", [])
    full_story_add = []
    story_content_add = []
    next_title_template = PromptTemplate.from_template("""
    You are a precise title selection assistant.

        Provided title: <{title_last_paragraph}>     
        Story structure <{story_structured_info}> 

        Rules:
        1. If the Provided title is <no title>, respond ONLY with the FIRST title from the Story structure.
        2. If the Provided title is the LAST title from the Story structure, respond ONLY with "FINISH".
        3. If the Provided title is any title in the middle of the Story structure, respond ONLY with the NEXT title immediately following the Provided title.

        Required checks:
        - if the Provided title is not <no title>, check the Provided title order number in the Story structure.
        - if the Provided title is not <no title>, check the title from the response order number in the Story structure.
        - if the Provided title is not <no title> and the order number of the title from the response is lower than order number of the Provided title, change response to "FINISH".
        - if the Provided title is not <no title> and the order number of the title from the response is equal to the order number of the Provided title, change response to "FINISH".
        - if the Provided title is the same as the last title from the Story structure, change response to "FINISH".


        IMPORTANT:                                                 
        DO NOT invent titles or provide titles that are before the Provided title in the Story Structure! Follow the Rules and Required checks!
        Respond only with the Selected title or with FINISH in plain text. 
    """)
    unit_length_template = PromptTemplate.from_template("""
    You are a very precise assistant.
    Please provide unit_length for the unit with the unit_name <{unit_name}> from the Story Structured Info <{story_structured_info}>. 
    If there is no unit_name or the unit_name is FINISH -> respond with 0.
    The unit_length should be a integer number. Respond in plain text.
    """)
    next_title_chain = next_title_template | openai_strict_llm
    unit_length_chain = unit_length_template | openai_llm
    if isinstance(actual_unit, list) and len(actual_unit) > 0:
        # summarize
        chapter_summarization_template = PromptTemplate.from_template("""
        You are a very talented writer.
        Please summarize the writing <{actual_unit}> in a few sentences (4 to 6). 
        Keep it clean and concise, but do not loose the main ideas from the writing. Keep the same language as the writting.
        Respond in plain text with the structure (in language of the writing): "Previous unit: summary of the writing"
        """)
        summarization_chain = chapter_summarization_template | openai_llm
        summarization_response = summarization_chain.invoke({"actual_unit": actual_unit})
        actual_summary = summarization_response.content
        last_paragraph = actual_unit[-1].content
        title_last_paragraph = actual_unit[0].content
        # add to full story
        actual_unit_text = [element.content for element in actual_unit] if len(actual_unit) > 0 else ""
        full_story_add = "\n".join(actual_unit_text) if isinstance(actual_unit_text, list) else actual_unit_text
        actual_unit = []
        # create story_content_add that will be appended as HummanMessage
        story_content_add.append('-------summary of the previous chapter-------')
        story_content_add.append(actual_summary)
        story_content_add.append('-------end of the summary-------')
        story_content_add.append("--------last paragraph from the previous chapter--------")
        story_content_add.append(last_paragraph)
        story_content_add.append("--------end of the last paragraph--------\n\n")
        response = next_title_chain.invoke({"title_last_paragraph": title_last_paragraph, "story_structured_info": state["story_structured_info"]})
    else:
        response = next_title_chain.invoke({"title_last_paragraph": "no title", "story_structured_info": state["story_structured_info"]})
    next_title = response.content
    unit_length_response = unit_length_chain.invoke({"unit_name": next_title, "story_structured_info": state["story_structured_info"]})
    unit_length = unit_length_response.content    
    story_content_add.append(next_title)
    story_content_add.append("\n")
    text_story_content = '\n'.join(story_content_add)

    if next_title == "" or next_title == "FINISH":
        return {"messages": AIMessage(content="FINISH"), "full_story": full_story_add, "story_content": story_content_add, "unit_length": str(unit_length)}

    actual_unit.append(AIMessage(content=next_title))
    story_content_add = [AIMessage(content=text_story_content)]
   
    return {"messages": AIMessage(content=next_title), "actual_unit": actual_unit, "full_story": full_story_add, "story_content": story_content_add, "unit_length": str(unit_length)}

def tools_router1(state: MainGraphState):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "END"

def chatbot_router6(
    state: MainGraphState,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if ai_message.content == "FINISH":
        return "story_saver"
    return "story_writer"

graph_builder = StateGraph(MainGraphState)
graph_builder.add_node("info_gatherer", inf_gatherer_subgraph)
graph_builder.add_node("character_supervisor", character_supervisor_subgraph)
graph_builder.add_node("story_structure_creator", story_structure_creator_subgraph)
graph_builder.add_node("story_writer", story_writer_subgraph)
graph_builder.add_node("structure_supervisor", structure_supervisor) 
graph_builder.add_node("story_saver", story_saver)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "info_gatherer")
graph_builder.add_edge("info_gatherer", "character_supervisor")
graph_builder.add_edge("character_supervisor", "story_structure_creator")
graph_builder.add_edge("story_structure_creator", "structure_supervisor")
graph_builder.add_edge("story_writer", "structure_supervisor")
graph_builder.add_conditional_edges(
    "story_saver",
    tools_router1,
    {"tools": "tools", "END": END},
)
graph_builder.add_conditional_edges(
    "structure_supervisor",
    chatbot_router6,
    {"story_writer": "story_writer", "story_saver": "story_saver"},
)
graph_builder.add_edge("tools", END)

graph = graph_builder.compile(checkpointer=memory)