from typing import Sequence, TypedDict,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,ToolMessage,BaseMessage,HumanMessage,AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode



import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int)-> int:
    """This function add two numbers together"""
    return a+b

tools = [add]


api_key = st.secrets["OPENAI_API_KEY"]
model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(tools)


def model_call(state:AgentState)-> AgentState:
    system_prompt = SystemMessage(content="You are my Ai Assistant answer my query with the best of your ability")
    
    response = model.invoke([system_prompt]+state["messages"])
    return {'messages':[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node('our_agent', model_call)

tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node )

graph.set_entry_point('our_agent')

graph.add_conditional_edges(
    'our_agent',
    should_continue, {
        'continue': 'tools',
        'end' : END
    }
)

graph.add_edge('tools', 'our_agent')

app = graph.compile()



def print_stream(stream):
    for s in stream:
        # s is a single value yielded by the stream
        if isinstance(s, tuple):
            role, content = s
            st.chat_message(role).markdown(content)
            print('this is 1')
        elif hasattr(s, "pretty_print"):
            st.chat_message('ai').markdown(s.pretty_print())
            print('this is 2')
        else:
            content = s.get('messages')[-1]
            print(type(content))
            if isinstance(content,HumanMessage):
                st.chat_message('human').markdown(content.content)    
                
            elif isinstance(content,AIMessage):
                if content.content == "":
                    st.chat_message('ai').markdown(f"Tool calling : " + str(content.additional_kwargs.get("tool_calls", [])))
                else:
                    st.chat_message('ai').markdown(content.content)    
            
                print('ai')

            else:
            
                print('tool')
                st.chat_message('assistant').markdown("Tool is running: " + str(content) )


st.title('React AI Agent')

prompt = st.chat_input('Write anything')

if prompt:
    #st.chat_message('user').markdown(prompt)
    input = {'messages':[('user', prompt)]}
    print_stream(app.stream(input,stream_mode="values"))



