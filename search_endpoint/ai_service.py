import json
import time
from typing import Dict, Any, Tuple , List
from fastapi import HTTPException
from google import genai
from google.genai import types

from settings import Settings as AppSettings
from schemas import AIResponse , ChatMessage
from logging_utils import StructuredLogger


#from llama_index.core.memory import ChatMemoryBuffer


class AIService:
    """AI service for generating responses using Gemini"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=AppSettings.GEMINI_API_KEY)
            self.logger.info("Gemini client initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Gemini client", e)
            self.client = None
    
    def generate_response(self, user_query: str, context: str) -> Tuple[AIResponse, Dict[str, Any], Dict[str, Any]]:
        """Generate structured response using Gemini with full logging"""
        
        if not self.client:
            raise HTTPException(status_code=503, detail="Gemini service not initialized")

        # Prepare input for logging
        gemini_input = {
            "model": AppSettings.GEMINI_MODEL,
            "system_instruction": self._prepare_system_prompt(context, user_query),
            "user_query": user_query,
            "config": {
                "temperature": AppSettings.GEMINI_TEMPERATURE,
                "max_output_tokens": AppSettings.GEMINI_MAX_OUTPUT_TOKENS,
                "top_p": AppSettings.GEMINI_TOP_P,
                "top_k": AppSettings.GEMINI_TOP_K,
                "response_mime_type": "application/json"
            }
        }

        try:
            # Make the structured API call
            response = self.client.models.generate_content(
                model=AppSettings.GEMINI_MODEL,
                contents=[user_query],
                config={
                    "system_instruction": gemini_input["system_instruction"],
                    "temperature": gemini_input["config"]["temperature"],
                    "max_output_tokens": gemini_input["config"]["max_output_tokens"],
                    "response_mime_type": "application/json",
                    "response_schema": AIResponse,  # Enforce our schema
                    "top_p": gemini_input["config"]["top_p"],
                    "top_k": gemini_input["config"]["top_k"]
                }
            )

            # Get both raw and parsed responses
            raw_response = response.text
            ai_response: AIResponse = response.parsed

            # Prepare output for logging
            gemini_output = {
                "raw_response": raw_response,
                "parsed_response": ai_response.dict(),
                "duration": response.metadata.generation_time if hasattr(response, 'metadata') else None
            }

            self.logger.info("Gemini response generated successfully", {
                "duration": gemini_output["duration"],
                "answer_length": len(ai_response.answer),
                "suggestions_count": len(ai_response.suggestions),
                "confidence_score": ai_response.confidence_score
            })

            return ai_response, gemini_input, gemini_output

        except Exception as e:
            self.logger.error("Gemini call failed", e)
            
            # Prepare error response
            error_response = AIResponse(
                answer=f"Error processing your request: {str(e)}",
                suggestions=["Please try again later", "Contact support if issue persists"],
                was_context_valid=False,
                confidence_score=0.0
            )
            
            return error_response, gemini_input, {
                "error": str(e),
                "raw_response": getattr(response, 'text', '') if 'response' in locals() else None
            }
    
    def _prepare_system_prompt(self, context: str, user_query: str) -> str:
        """Prepare enhanced system prompt with complete metadata structure"""
        return f"""You are an expert document analysis assistant. Answer the user's question using ONLY the provided context below.

            IMPORTANT INSTRUCTIONS:
            1. Each document chunk has inline metadata in [METADATA]...[/METADATA] blocks.
            2. The metadata may include:
            - DOC_TITLE: Title of the document
            - DOC_DESCRIPTION: Short summary or description of the content
            - DOC_DESCRIPTION_FORMATTED: Richly formatted version of the description
            - TAGS: Keywords associated with the document
            - PRESENTATION_DATE: Date when the presentation or document was created/shared
            - DOC_MODULE: Related module or functional area
            - PRESENTATION_LINK: Link to the full presentation (often a Google Drive URL)
            - PRESENTER_1_NAME: Name of the person in case of ppt the presentor , in case of pdf the person who shares the pdf.
            - SCORE: Relevance score (use this to prioritize information)
            - DOC_REF_ID: Reference ID used to cite the document
            3. Always prioritize content with a higher SCORE.
            4. When referencing documents in your answer, mention the PRESENTATION_LINK if helpful.
            5. Use the DOC_TITLE and PRESENTER_1_NAME   only if the answer truly requires deeper exploration.
            6. When presenting suggestions, base them on what’s in context, not outside assumptions.

            CONTEXT WITH INLINE METADATA:
            {context}

            RESPONSE FORMAT (JSON):
            {{
                "answer": "Comprehensive answer according to the context and inline metadata provided to you with source references using PRESENTATION_LINK",
                "suggestions": [
                    "Follow-up question 1 based on available content",
                    "Follow-up question 2 about related topics",
                    "Follow-up question 3 building on current answer"
                ],
                "was_context_valid": true/false (based on whether context fully answers the question),
                "confidence_score": 0.0–1.0 (based on relevance scores and content quality)
            }}

            USER QUESTION: {user_query}

            Respond with ONLY the JSON object:"""

    
    
    def _prepare_system_prompt_chat(self, context: str, user_query: str) -> str:
        """Prepare enhanced system prompt with complete metadata structure"""
        return f"""You are an AI Knowledge Assistant* – friendly, helpful, and specialized in document and policy understanding.

Your responsibilities:
- Help employees by answering questions using internal documents (PDFs, PPTs, training decks, etc.).
- Respond to greetings, feedback, and general queries in a human, polite, and assistant-like manner.
- When asked a question that requires information, try to answer based on the provided context.

BEHAVIOR INSTRUCTIONS:
------------------------------
1. If the user is greeting (e.g., "hello", "good morning"), respond warmly, ignoring the document context.
2. If the user says thanks or sends feedback, reply nicely, again without using context.
3. For all other questions, use the context provided below.
4. If the provided context doesn't help answer the question, return `"was_context_valid": false` — we will retrieve new context for you.
5. Never fabricate answers. Be polite and say you don't know *if* the context is insufficient.
6. You MUST always use the JSON format shown below to respond.

CONTEXT STRUCTURE & METADATA GUIDE:
----------------------------------------------
Each document chunk comes with inline metadata formatted as:
[METADATA]
  DOC_TITLE: ...
  DOC_DESCRIPTION: ...
  DOC_DESCRIPTION_FORMATTED: ...
  TAGS: ...
  PRESENTATION_DATE: ...
  DOC_MODULE: ...
  PRESENTATION_LINK: ...
  PRESENTER_1_NAME: ...
  SCORE: ...
  DOC_REF_ID: ...
[/METADATA]

Instructions for using metadata:
1. Prioritize chunks with a higher `SCORE`.
2. If available, include `PRESENTATION_LINK` in the answer.
3. Mention `DOC_TITLE` or `PRESENTER_1_NAME` only if it improves clarity.
4. NEVER use any info outside this context. Do not guess or hallucinate.
5. Your suggestions should always be based on content from the context.

PROVIDED CONTEXT:
------------------------------
{context}

RESPONSE FORMAT (strict JSON):
{{
  "answer": 
  \"\"\"Your full answer here in MARKDOWN format:
  
  - For greetings: Friendly, conversational response
  - For document queries: Detailed response with proper formatting
  - Format drive links as: For more details, check [here](PRESENTATION_LINK)
  - Use bullet points (- or *) for lists
  - Use headers (##) for sections when needed
  - Example greeting: \"Hello! How can I assist you today?\"
  - Example document response:
    \"\"\"
    Here's the information you requested:

    - Main Point: Explanation of concept
    - Key Finding: Important discovery
    - For more details, check out [here](https://drive.google.com/...)
    
    Additional details:
    Supporting detail 1
    Supporting detail 2
    \"\"\"
  \"\"\",
  "suggestions": [
    "Follow-up question suggestion 1",
    "Suggestion 2 based on current content",
    "Suggestion 3 if applicable"
  ],
  "was_context_valid": true or false,
  "confidence_score": 0.0-1.0,
  
  IMPORTANT:
  - For greetings/casual conversation: 
    * Set "was_context_valid": true (no context needed)
    * Keep answer friendly and conversational
  - For document queries:
    * Set "was_context_valid": true ONLY if context fully answers the question
    * Use proper Markdown formatting
    * Include PRESENTATION_LINK when available
    * Maintain detailed, structured responses
  - Always:
    * Return ONLY the JSON object
    * Maintain all existing functionality
    * Keep suggestions relevant to content
}}

USER QUESTION: {user_query}

Respond with ONLY the JSON object:
"""

    def generate_chat_response(self, message_history: List[ChatMessage], current_question: str,
                           context: str) -> Tuple[AIResponse, Dict[str, Any], Dict[str, Any]]:
        """Generate structured response using Gemini with chat history support"""

        if not self.client:
            raise HTTPException(status_code=503, detail="Gemini service not initialized")

        import time
        import json

        start_time = time.time()

        # Prepare system prompt
        system_prompt = self._prepare_system_prompt_chat(context, current_question)

        # Build message history with role and parts dicts
        gemini_contents = []
        for msg in message_history:
            if msg.role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "model":
                gemini_contents.append({"role": "model", "parts": [{"text": msg.content}]})

        # Add the current question as the final user input
        gemini_contents.append({"role": "user", "parts": [{"text": current_question}]})

        # Prepare input for logging
        gemini_input = {
            "model": AppSettings.GEMINI_MODEL,
            "system_instruction": system_prompt,
            "contents": gemini_contents,
            "message_history_length": len(message_history),
            "current_question": current_question,
            "config": {
                "temperature": AppSettings.GEMINI_TEMPERATURE,
                "max_output_tokens": AppSettings.GEMINI_MAX_OUTPUT_TOKENS,
                "top_p": AppSettings.GEMINI_TOP_P,
                "top_k": AppSettings.GEMINI_TOP_K,
                "response_mime_type": "application/json"
            }
        }

        try:
            # Call Gemini API
            response = self.client.models.generate_content(
                model=AppSettings.GEMINI_MODEL,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=AppSettings.GEMINI_TEMPERATURE,
                    max_output_tokens=AppSettings.GEMINI_MAX_OUTPUT_TOKENS,
                    top_p=AppSettings.GEMINI_TOP_P,
                    top_k=AppSettings.GEMINI_TOP_K,
                    response_mime_type="application/json"
                )
            )

            duration = time.time() - start_time

            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Try to parse structured JSON response
            response_data = json.loads(response.text)

            gemini_output = {
                "raw_response": response.text,
                "parsed_response": response_data,
                "duration": duration
            }

            structured_response = AIResponse(
                answer=response_data.get("answer", "No answer provided"),
                suggestions=response_data.get("suggestions", []),
                was_context_valid=response_data.get("was_context_valid", False),
                confidence_score=min(max(response_data.get("confidence_score", 0.0), 0.0), 1.0)
            )

            self.logger.info("Gemini chat response generated successfully", {
                "duration": duration,
                "answer_length": len(structured_response.answer),
                "suggestions_count": len(structured_response.suggestions),
                "confidence_score": structured_response.confidence_score,
                "message_history_length": len(message_history)
            })

            return structured_response, gemini_input, gemini_output

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse Gemini response", e)
            gemini_output = {"error": str(e), "raw_response": getattr(response, 'text', '')}
            fallback_response = AIResponse(
                answer="Failed to parse AI response - please try rephrasing your question",
                suggestions=["Try asking about specific topics", "Rephrase with more specific keywords"],
                was_context_valid=False,
                confidence_score=0.0
            )
            return fallback_response, gemini_input, gemini_output

        except Exception as e:
            self.logger.error("Gemini chat call failed", e)
            gemini_output = {"error": str(e)}
            fallback_response = AIResponse(
                answer=f"Error processing your request: {str(e)}",
                suggestions=["Please try again later", "Contact support if issue persists"],
                was_context_valid=False,
                confidence_score=0.0
            )
            return fallback_response, gemini_input, gemini_output

        
        