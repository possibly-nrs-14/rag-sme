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

                IMPORTANT RULES:
                1. When user asks to "generate quiz AND export", you MUST use TWO actions in sequence:
                - First: generate_quiz_on_neuroanatomy_topic to create the quiz
                - Second: export_document to export it as PDF/DOCX/PPTX
                2. When user asks to "answer question AND export", use answer_question_about_neuroanatomy then export_document
                3. For export_document, the system will automatically use the quiz/QA from the PREVIOUS step.
                   Your Action Input MUST be a SIMPLE JSON object with ONLY "export_type" and "file_format".
                   - Example 1 (for exporting a quiz to PDF): {{"export_type": "quiz", "file_format": "pdf"}}
                   - Example 2 (for exporting an answer to DOCX): {{"export_type": "qa", "file_format": "docx"}}
                   - If file_format is not specified by user, default to "pdf".
                   - DO NOT include a "data" field in your Action Input. The system handles it automatically.

                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action as JSON
                Observation: The result of the action
                ... (this Thought/Action/Observation can repeat N times)

                When you have a final response for the user:

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

                IMPORTANT RULES:
                1. When user asks to "generate quiz AND export", you MUST use TWO actions in sequence:
                   - First: generate_quiz_on_neuroanatomy_topic to create the quiz
                   - Second: export_document to export it as PDF/DOCX/PPTX
                2. When user asks to "answer question AND export", use answer_question_about_neuroanatomy then export_document
                3. For export_document, the system will automatically use the quiz/QA from the PREVIOUS step.
                   Your Action Input MUST be a SIMPLE JSON object with ONLY "export_type" and "file_format".
                   - Example 1 (for exporting a quiz to PDF): {{"export_type": "quiz", "file_format": "pdf"}}
                   - Example 2 (for exporting an answer to DOCX): {{"export_type": "qa", "file_format": "docx"}}
                   - If file_format is not specified by user, default to "pdf".
                   - DO NOT include a "data" field in your Action Input. The system handles it automatically.
                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action as JSON
                Observation: The result of the action
                ... (this Thought/Action/Observation can repeat N times)

                When you have a final response for the user:

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

                IMPORTANT RULES:
                1. When user asks to "generate quiz AND export", you MUST use TWO actions in sequence:
                   - First: generate_quiz_on_neuroanatomy_topic to create the quiz
                   - Second: export_document to export it as PDF/DOCX/PPTX
                2. When user asks to "answer question AND export", use answer_question_about_neuroanatomy then export_document
                3. For export_document, the system will automatically use the quiz/QA from the PREVIOUS step.
                   Your Action Input MUST be a SIMPLE JSON object with ONLY "export_type" and "file_format".
                   - Example 1 (for exporting a quiz to PDF): {{"export_type": "quiz", "file_format": "pdf"}}
                   - Example 2 (for exporting an answer to DOCX): {{"export_type": "qa", "file_format": "docx"}}
                   - If file_format is not specified by user, default to "pdf".
                   - DO NOT include a "data" field in your Action Input. The system handles it automatically.
                To use a tool, please use the following format:

                Thought: Do I need to use a tool? Yes
                Action: The action to take, should be one of [{tool_names}]
                Action Input: The input to the action as JSON
                Observation: The result of the action
                ... (this Thought/Action/Observation can repeat N times)

                When you have a final response for the user:

                Thought: Do I need to use a tool? No
                Final Answer: [your response to the user]

                Previous conversation history:
                {chat_history}

                New input: {user_input}
                {agent_scratchpad}"""
                        
        return prompt
    
    def _parse_action(self, text):
        # 1. Try to parse an Action first
        action_pattern = r"Action:\s*(.+?)(?:\n|$)"
        action_input_pattern = r"Action Input:\s*(.+?)(?:\n(?:Observation|Thought|Action|Final Answer)|$)"

        action_match = re.search(action_pattern, text, re.IGNORECASE)
        action_input_match = re.search(action_input_pattern, text, re.IGNORECASE | re.DOTALL)

        if action_match:
            action = action_match.group(1).strip()
            action_input_raw = action_input_match.group(1).strip() if action_input_match else ""

            # Try to parse action_input as JSON, otherwise keep as string
            try:
                action_input = json.loads(action_input_raw)
            except Exception:
                action_input = action_input_raw

            return {
                "action": action,
                "action_input": action_input
            }

        # 2. If no Action, try Final Answer
        final_answer_pattern = r"Final Answer:\s*(.+?)(?:\n\n|$)"
        final_match = re.search(final_answer_pattern, text, re.IGNORECASE | re.DOTALL)
        if final_match:
            return {"final_answer": final_match.group(1).strip()}

        return None

    def _normalize_export_input(self, action_input, intermediate_steps):
        """
        Repair and complete the payload for ExportTool.
        - Accepts loose inputs from the LLM.
        - Pulls the last QA/quiz observation from intermediate_steps.
        """
        # Normalize to dict
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except Exception:
                action_input = {"data": action_input}
        if not isinstance(action_input, dict):
            action_input = {}

        export_type = action_input.get("export_type")
        file_format = action_input.get("file_format") or "pdf"
        data = action_input.get("data")

        # Look backwards for last structured observation (QA or quiz)
        last_struct = None
        last_kind = None      # "qa" or "quiz"
        last_question = None
        last_topic = None

        for step in reversed(intermediate_steps):
            obs = step.get("observation")
            act = step.get("action")

            # Capture QA question
            if act == "answer_question_about_neuroanatomy":
                ai = step.get("action_input") or {}
                if isinstance(ai, dict):
                    q = ai.get("question")
                    if q:
                        last_question = q

            # Capture quiz topic
            if act == "generate_quiz_on_neuroanatomy_topic":
                ai = step.get("action_input") or {}
                if isinstance(ai, dict):
                    t = ai.get("topic")
                    if t:
                        last_topic = t

            if isinstance(obs, dict):
                if "items" in obs:
                    last_struct = obs
                    last_kind = "quiz"
                elif "answer" in obs:
                    last_struct = obs
                    last_kind = "qa"

            if last_struct and last_kind == "quiz" and last_topic:
                break
            if last_struct and last_kind == "qa" and last_question:
                break

        # If we have a structured result, build proper payload
        if last_struct is not None:
            if not export_type:
                export_type = last_kind

            if not isinstance(data, dict):
                data = dict(last_struct)

            if last_kind == "qa":
                if "question" not in data and last_question:
                    data["question"] = last_question
                if "sources" not in data:
                    data["sources"] = last_struct.get("sources", []) or []

            elif last_kind == "quiz":
                if "topic" not in data:
                    if "topic" in last_struct:
                        data["topic"] = last_struct["topic"]
                    elif last_topic:
                        data["topic"] = last_topic
                if "sources" not in data:
                    data["sources"] = last_struct.get("sources", []) or []

            return {
                "export_type": export_type,
                "file_format": file_format,
                "data": data,
            }

        # Fallback: treat as simple QA text export
        answer_text = ""
        if isinstance(data, str):
            answer_text = data
        elif isinstance(action_input.get("data"), str):
            answer_text = action_input["data"]

        return {
            "export_type": export_type or "qa",
            "file_format": file_format,
            "data": {
                "question": "",
                "answer": answer_text,
                "sources": []
            },
        }

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
            
            thought = parsed.get("thought", "")
            action = parsed["action"]
            action_input = parsed["action_input"]

            # Special handling for export_document: repair/complete the payload
            if action == "export_document":
                action_input = self._normalize_export_input(action_input, intermediate_steps)   

            if self.verbose:
                if thought:
                    print(f"Thought: {thought}")
                print(f"→ Executing tool: {action}")
                print(f"→ Tool input: {action_input}\n")

            observation_obj = self._execute_tool(action, action_input)
            # observation_str = str(observation_obj)

            observation_for_scratchpad = ""
            if action == "generate_quiz_on_neuroanatomy_topic":
                if isinstance(observation_obj, dict) and observation_obj.get("items"):
                    item_count = len(observation_obj['items'])
                    observation_for_scratchpad = f"Successfully generated {item_count} quiz questions."
                    if self.verbose:
                         print(f"← Observation (Full): {str(observation_obj)[:300]}...")
                         print(f"← Observation (To Scratchpad): {observation_for_scratchpad}\n")
                else:
                    observation_for_scratchpad = "Quiz generation tool ran, but returned an unexpected result or error."
            elif action == "answer_question_about_neuroanatomy":
                 if isinstance(observation_obj, dict) and observation_obj.get("answer"):
                    observation_for_scratchpad = "Successfully found an answer for the question."
                    if self.verbose:
                         print(f"← Observation (Full): {str(observation_obj)[:300]}...")
                         print(f"← Observation (To Scratchpad): {observation_for_scratchpad}\n")
                 else:
                    observation_for_scratchpad = "QA tool ran, but returned an unexpected result or error."
            else:
                # For simple tools (like export_document), the string output is fine.
                observation_for_scratchpad = str(observation_obj)
                if self.verbose:
                    obs_preview = observation_for_scratchpad[:300] + "..." if len(observation_for_scratchpad) > 300 else observation_for_scratchpad
                    print(f"← Observation: {obs_preview}\n")

            # if self.verbose:
            #     obs_preview = observation_str[:300] + "..." if len(observation_str) > 300 else observation_str
            #     print(f"← Observation: {obs_preview}\n")

            intermediate_steps.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation_obj
            })
            
            agent_scratchpad += f"\nThought: I need to use a tool\n"
            agent_scratchpad += f"Action: {action}\n"
            agent_scratchpad += f"Action Input: {json.dumps(action_input) if isinstance(action_input, dict) else action_input}\n"
            agent_scratchpad += f"Observation: {observation_for_scratchpad}\n" 
        
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