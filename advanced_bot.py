"""
Advanced LangChain Agentic AI Bot with Memory and Custom Personalities
Uses Groq API for fast, free inference
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class AdvancedAgenticBot:
    """Advanced bot with memory and customization"""
    
    def __init__(self, personality: str = "helpful"):
        self.llm = self._initialize_llm()
        self.personality = personality
        self.conversation_history = []
        self.memory_file = "bot_memory.json"
        self.load_memory()
        
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,
            groq_api_key=api_key,
            max_tokens=1024
        )
    
    @tool
    def calculator(self, expression: str) -> str:
        """Perform mathematical calculations"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def web_search(self, query: str) -> str:
        """Search for information online"""
        search = DuckDuckGoSearchRun()
        try:
            return search.run(query)
        except Exception as e:
            return f"Could not search: {str(e)}"
    
    @tool
    def store_memory(self, key: str, value: str) -> str:
        """Store information in bot memory"""
        memory = self._load_memory_file()
        memory[key] = value
        self._save_memory_file(memory)
        return f"Stored: {key} = {value}"
    
    @tool
    def recall_memory(self, key: str) -> str:
        """Recall stored information"""
        memory = self._load_memory_file()
        if key in memory:
            return f"{key}: {memory[key]}"
        return f"No memory found for '{key}'"
    
    def get_personality_prompt(self) -> str:
        """Get system prompt based on personality"""
        personalities = {
            "helpful": "You are a helpful and friendly AI assistant.",
            "expert": "You are an expert AI advisor with deep knowledge.",
            "casual": "You are a casual and fun AI buddy who keeps things light.",
            "technical": "You are a technical expert who provides detailed, precise answers."
        }
        return personalities.get(self.personality, personalities["helpful"])
    
    def create_agent(self):
        """Create the agent with tools"""
        tools = [
            self.calculator,
            self.web_search,
            self.store_memory,
            self.recall_memory
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_personality_prompt() + 
             "\nUse your tools wisely. Remember previous context from our conversation."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    def load_memory(self):
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {}
    
    def _load_memory_file(self) -> Dict:
        """Load memory from disk"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_memory_file(self, memory: Dict):
        """Save memory to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump(memory, f, indent=2)
    
    def format_message(self, sender: str, message: str) -> str:
        """Format message for display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {sender}: {message}"
    
    def run(self):
        """Run the advanced bot"""
        print("\n" + "="*60)
        print("Advanced LangChain Agentic AI Bot")
        print("="*60)
        print(f"Personality: {self.personality.upper()}")
        print("\nCommands:")
        print("  memory list   - Show all stored memories")
        print("  personality   - Change personality")
        print("  help          - Show available commands")
        print("  exit/quit     - End conversation\n")
        
        agent_executor = self.create_agent()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == "memory list":
                    memory = self._load_memory_file()
                    if memory:
                        print("\nStored Memories:")
                        for k, v in memory.items():
                            print(f"  • {k}: {v}")
                    else:
                        print("No memories stored yet.")
                    continue
                
                if user_input.lower() == "personality":
                    print("Available personalities: helpful, expert, casual, technical")
                    new_personality = input("Choose personality: ").strip().lower()
                    if new_personality in ["helpful", "expert", "casual", "technical"]:
                        self.personality = new_personality
                        print(f"Personality changed to {new_personality}")
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\nAssistant: Thanks for chatting! Goodbye!")
                    break
                
                # Process normal conversation
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": self.conversation_history
                })
                
                assistant_message = response.get("output", "I couldn't generate a response.")
                print(f"\nAssistant: {assistant_message}")
                
                self.conversation_history.append(HumanMessage(content=user_input))
                self.conversation_history.append(AIMessage(content=assistant_message))
            
            except KeyboardInterrupt:
                print("\n\nBot interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")

def main():
    try:
        # Choose personality
        print("\n" + "="*60)
        print("Welcome to Advanced LangChain Agentic Bot!")
        print("="*60)
        print("\nChoose your bot's personality:")
        print("  1. helpful   - Friendly and helpful")
        print("  2. expert    - Technical and precise")
        print("  3. casual    - Fun and informal")
        print("  4. technical - Deep technical knowledge")
        
        choice = input("\nEnter choice (1-4) or personality name: ").strip().lower()
        
        personality_map = {"1": "helpful", "2": "expert", "3": "casual", "4": "technical"}
        personality = personality_map.get(choice, choice) if choice in personality_map else choice
        
        if personality not in ["helpful", "expert", "casual", "technical"]:
            personality = "helpful"
        
        # Run bot
        bot = AdvancedAgenticBot(personality=personality)
        bot.run()
    
    except ValueError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
