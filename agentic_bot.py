"""
LangChain Agentic AI Bot using Groq API
Asks questions to the user and returns answers
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize the Groq LLM
def initialize_groq_llm():
    """Initialize Groq LLM with free model (mixtral-8x7b-32768)"""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file")
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  # Free model from Groq
        temperature=0.7,
        groq_api_key=api_key,
        max_tokens=1024
    )
    return llm

# Define custom tools for the agent
@tool
def calculator(expression: str) -> str:
    """
    Useful for performing mathematical calculations.
    Input should be a mathematical expression.
    """
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def get_current_information(query: str) -> str:
    """
    Search for current information from the web.
    Use this when you need real-time information or facts.
    """
    search = DuckDuckGoSearchRun()
    try:
        result = search.run(query)
        return result
    except Exception as e:
        return f"Could not fetch information: {str(e)}"

def create_agent():
    """Create and initialize the agentic bot"""
    
    # Initialize LLM
    llm = initialize_groq_llm()
    
    # Define tools
    tools = [
        calculator,
        get_current_information,
        Tool(
            name="Memory",
            func=lambda x: x,
            description="Use this to remember information shared by the user"
        )
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and intelligent AI assistant. 
You ask clarifying questions when needed and provide detailed, accurate answers.
Be conversational and friendly. When you don't know something, admit it and offer to search for information.
Use the available tools to help answer user queries effectively."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

def main():
    """Main function to run the agentic bot"""
    
    print("=" * 60)
    print("Welcome to LangChain Agentic AI Bot powered by Groq!")
    print("=" * 60)
    print("\nThis bot can:")
    print("  • Answer your questions")
    print("  • Perform calculations")
    print("  • Search for current information")
    print("  • Have intelligent conversations")
    print("\nType 'exit' or 'quit' to end the conversation.\n")
    
    try:
        # Create the agent
        agent_executor = create_agent()
        
        # Conversation loop
        conversation_history = []
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAssistant: Thank you for chatting with me! Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                print("Please enter a question or message.")
                continue
            
            try:
                # Run the agent
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": conversation_history
                })
                
                # Extract and display the response
                assistant_message = response.get("output", "I couldn't generate a response.")
                print(f"\nAssistant: {assistant_message}")
                
                # Store in conversation history
                conversation_history.append(("human", user_input))
                conversation_history.append(("assistant", assistant_message))
                
            except Exception as e:
                print(f"\nAssistant: I encountered an error: {str(e)}")
                print("Please try again with a different question.")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease follow these steps:")
        print("1. Get a free Groq API key from https://console.groq.com/keys")
        print("2. Create a .env file in this directory")
        print("3. Add your API key: GROQ_API_KEY=your_key_here")
    
    except KeyboardInterrupt:
        print("\n\nBot interrupted by user. Goodbye!")

if __name__ == "__main__":
    main()
