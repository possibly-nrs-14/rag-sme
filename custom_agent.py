import re
import json
import logging

logger = logging.getLogger(__name__)


class AgentExecutor:
    
    def __init__(self, llm, model_name, tools, max_iterations=5, verbose=True):
        self.llm = llm
        self.model_name=model_name
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def _format_tools(self):
        tool_strings = []
        for name, tool in self.tools.items():
            tool_strings.append(f"- {name}: {tool.description}")
        return "\n".join(tool_strings)
    
    def _format_tool_names(self):
        return ", ".join(self.tools.keys())
    
    def _build_prompt(self, user_input, chat_history, agent_scratchpad):
        tools_desc = self._format_tools()
        tool_names = self._format_tool_names()
        
        if self.model_name == 'microsoft/MediPhi':
            prompt = f"""<|system|>
                You are a helpful Subject Matter Expert AI Agent specializing in neuroanatomy.
                You have access to a set of tools to answer questions, generate quizzes, and export documents.

                TOOLS:
                ------
                {tools_desc}

                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action
                Observation: The result of the action

                When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

                Thought: Do I need to use a tool? No
                Final Answer: [your response to the user]
                <|end|>
                <|user|>
                Previous conversation history:
                {chat_history}

                New input: {user_input}
                {agent_scratchpad}
                <|end|>
                <|assistant|>"""

        elif self.model_name == 'Intelligent-Internet/II-Medical-8B':
            prompt = f"""<|system|>
                You are a helpful Subject Matter Expert AI Agent specializing in neuroanatomy.
                You have access to a set of tools to answer questions, generate quizzes, and export documents.

                TOOLS:
                ------
                {tools_desc}

                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action
                Observation: The result of the action

                When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

                Thought: Do I need to use a tool? No
                Final Answer: [your response to the user]
                <|end|>
                <|user|>
                Previous conversation history:
                {chat_history}

                New input: {user_input}
                {agent_scratchpad}
                <|end|>
                <|assistant|>"""
                        
        else:
            prompt = f"""You are a helpful Subject Matter Expert AI Agent specializing in neuroanatomy.
                You have access to a set of tools to answer questions, generate quizzes, and export documents.

                TOOLS:
                ------
                {tools_desc}

                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action
                Observation: The result of the action

                When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

                Thought: Do I need to use a tool? No
                Final Answer: [your response to the user]

                Previous conversation history:
                {chat_history}

                New input: {user_input}
                {agent_scratchpad}"""
                        
        return prompt
    
    def _parse_action(self, text):
        final_answer_pattern = r"Final Answer:\s*(.+?)(?:\n\n|$)"
        final_match = re.search(final_answer_pattern, text, re.IGNORECASE | re.DOTALL)
        if final_match:
            return {"final_answer": final_match.group(1).strip()}
        
        action_pattern = r"Action:\s*(.+?)(?:\n|$)"
        action_input_pattern = r"Action Input:\s*(.+?)(?:\n(?:Observation|Thought|Action)|$)"
        
        action_match = re.search(action_pattern, text, re.IGNORECASE)
        action_input_match = re.search(action_input_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip() if action_input_match else ""
            
            try:
                action_input_clean = action_input.strip()
                if action_input_clean.startswith('```json'):
                    action_input_clean = action_input_clean[7:]
                if action_input_clean.startswith('```'):
                    action_input_clean = action_input_clean[3:]
                if action_input_clean.endswith('```'):
                    action_input_clean = action_input_clean[:-3]
                action_input_clean = action_input_clean.strip()
                
                action_input = json.loads(action_input_clean)
            except:
                pass
            
            return {
                "action": action,
                "action_input": action_input
            }
        
        return None
    
    def _execute_tool(self, tool_name, tool_input):
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = tool.invoke(tool_input)
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool: {str(e)}"
    
    def invoke(self, inputs):
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", "")
        
        intermediate_steps = []
        agent_scratchpad = ""
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*60}")
            
            prompt = self._build_prompt(
                user_input=user_input,
                chat_history=chat_history,
                agent_scratchpad=agent_scratchpad,
            )
            
            try:
                response = self.llm.invoke(prompt)
                
                if isinstance(response, dict):
                    response = response.get('text', str(response))
                response = str(response).strip()
                
            except Exception as e:
                logger.error(f"Error invoking LLM: {e}")
                return {
                    "output": f"Error: Failed to get response from model - {str(e)}",
                    "intermediate_steps": intermediate_steps
                }
            
            if self.verbose:
                print(f"\nAgent response:\n{response}\n")
            
            parsed = self._parse_action(response)
            
            if not parsed:
                if "Final Answer:" in response:
                    answer = response.split("Final Answer:")[-1].strip()
                    return {
                        "output": answer,
                        "intermediate_steps": intermediate_steps
                    }
                return {
                    "output": response,
                    "intermediate_steps": intermediate_steps
                }
            
            if "final_answer" in parsed:
                return {
                    "output": parsed["final_answer"],
                    "intermediate_steps": intermediate_steps
                }
            
            action = parsed["action"]
            action_input = parsed["action_input"]
            
            if self.verbose:
                print(f"→ Executing tool: {action}")
                print(f"→ Tool input: {action_input}\n")
            
            observation_obj = self._execute_tool(action, action_input)
            observation_str = str(observation_obj)
            
            if self.verbose:
                obs_preview = observation_str[:300] + "..." if len(observation_str) > 300 else observation_str
                print(f"← Observation: {obs_preview}\n")
            
            intermediate_steps.append({
                "action": action,
                "action_input": action_input,
                "observation": observation_obj
            })
            
            agent_scratchpad += f"\nThought: I need to use a tool\n"
            agent_scratchpad += f"Action: {action}\n"
            agent_scratchpad += f"Action Input: {json.dumps(action_input) if isinstance(action_input, dict) else action_input}\n"
            agent_scratchpad += f"Observation: {observation_str}\n"
        
        return {
            "output": "I apologize, but I couldn't complete the task within the allowed steps. Please try breaking down your request or rephrasing it.",
            "intermediate_steps": intermediate_steps
        }


class ConversationMemory:
    
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
    
    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
    
    def get_history(self):
        return self.history.copy()
    
    def get_history_string(self):
        if not self.history:
            return "No previous conversation."
        
        history_str = ""
        for msg in self.history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        return history_str.strip()
    
    def clear(self):
        self.history = []
        logger.info("Conversation history cleared.")