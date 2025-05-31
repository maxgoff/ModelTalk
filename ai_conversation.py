#!/usr/bin/env python3
"""
AI Model Conversation Script
A configurable tool for orchestrating conversations between different AI models
from various providers (Anthropic, OpenAI, Google, Meta, Ollama).
"""

import asyncio
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import argparse
from pathlib import Path

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    OLLAMA = "ollama"

class StopCondition(Enum):
    FIXED_TURNS = "fixed_turns"
    TIME_LIMIT = "time_limit"
    MODEL_DECIDES = "model_decides"

@dataclass
class ModelConfig:
    provider: Provider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7

@dataclass
class ConversationConfig:
    stop_condition: StopCondition
    max_turns: Optional[int] = None
    time_limit_minutes: Optional[float] = None
    starting_question: str = ""
    save_to_file: bool = True
    output_dir: str = "conversations"

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float
    model_info: Dict[str, Any]
    turn_number: int

class AIModelConversation:
    def __init__(self, model1_config: ModelConfig, model2_config: ModelConfig, 
                 conversation_config: ConversationConfig):
        self.model1 = model1_config
        self.model2 = model2_config
        self.config = conversation_config
        self.conversation_history: List[Message] = []
        self.start_time = None
        self.conversation_file_path = None
        
    async def _make_api_call(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Make API call to the specified model provider."""
        
        if model.provider == Provider.ANTHROPIC:
            return await self._call_anthropic(model, messages, session)
        elif model.provider == Provider.OPENAI:
            return await self._call_openai(model, messages, session)
        elif model.provider == Provider.GOOGLE:
            return await self._call_google(model, messages, session)
        elif model.provider == Provider.META:
            return await self._call_meta(model, messages, session)
        elif model.provider == Provider.OLLAMA:
            return await self._call_ollama(model, messages, session)
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    async def _call_anthropic(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Call Anthropic Claude API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": model.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Convert messages format for Anthropic
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append(msg)
        
        payload = {
            "model": model.model_name,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "messages": claude_messages
        }
        
        if system_message:
            payload["system"] = system_message
            
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["content"][0]["text"]
    
    async def _call_openai(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Call OpenAI API."""
        url = model.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature
        }
        
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_google(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Call Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.model_name}:generateContent?key={model.api_key}"
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append({"parts": [{"text": msg["content"]}], "role": "user"})
            elif msg["role"] == "assistant":
                contents.append({"parts": [{"text": msg["content"]}], "role": "model"})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": model.temperature,
                "maxOutputTokens": model.max_tokens
            }
        }
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Google API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
    
    async def _call_meta(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Call Meta Llama API (via together.ai or similar)."""
        # This assumes using together.ai or similar service for Meta models
        url = model.base_url or "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature
        }
        
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Meta API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_ollama(self, model: ModelConfig, messages: List[Dict], session: aiohttp.ClientSession) -> str:
        """Call Ollama local API."""
        base_url = model.base_url or "http://localhost:11434"
        url = f"{base_url}/api/chat"
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens
            }
        }
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error: {response.status} - {error_text}")
            
            result = await response.json()
            return result["message"]["content"]
    
    def _should_continue(self, turn_number: int) -> bool:
        """Check if conversation should continue based on stop condition."""
        if self.config.stop_condition == StopCondition.FIXED_TURNS:
            return turn_number < self.config.max_turns
        elif self.config.stop_condition == StopCondition.TIME_LIMIT:
            elapsed = (time.time() - self.start_time) / 60  # minutes
            return elapsed < self.config.time_limit_minutes
        elif self.config.stop_condition == StopCondition.MODEL_DECIDES:
            # Check if either model indicated they want to end
            if len(self.conversation_history) >= 2:
                last_message = self.conversation_history[-1].content.lower()
                end_phrases = ["goodbye", "end conversation", "that concludes", "farewell", 
                              "this has been", "thank you for the conversation"]
                return not any(phrase in last_message for phrase in end_phrases)
            return True
        return False
    
    def _create_system_message(self, model: ModelConfig, other_model: ModelConfig, is_first: bool) -> str:
        """Create system message for model introduction."""
        return f"""You are about to engage in a conversation with another AI model. 

Your details:
- Provider: {model.provider.value}
- Model: {model.model_name}

Other model details:
- Provider: {other_model.provider.value}  
- Model: {other_model.model_name}

You are {"starting" if is_first else "joining"} this conversation. Please introduce yourself briefly and naturally, mentioning your name/model if you'd like. Then engage with the topic and the other AI model in a natural, conversational way.

The conversation will continue until a stopping condition is met. Feel free to ask questions, share insights, and engage authentically."""
    
    async def _initialize_conversation_file(self):
        """Initialize the conversation JSON file."""
        if self.config.save_to_file:
            os.makedirs(self.config.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{self.model1.provider.value}_{self.model2.provider.value}_{timestamp}.json"
            self.conversation_file_path = Path(self.config.output_dir) / filename
            
            # Create initial file structure
            initial_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model1": self._serialize_model_config(self.model1),
                    "model2": self._serialize_model_config(self.model2),
                    "config": self._serialize_conversation_config(self.config),
                    "status": "in_progress"
                },
                "conversation": []
            }
            
            with open(self.conversation_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Conversation file created: {self.conversation_file_path}")
    
    def _serialize_model_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Serialize ModelConfig to JSON-compatible dict."""
        data = asdict(config)
        data["provider"] = config.provider.value
        return data
    
    def _serialize_conversation_config(self, config: ConversationConfig) -> Dict[str, Any]:
        """Serialize ConversationConfig to JSON-compatible dict."""
        data = asdict(config)
        data["stop_condition"] = config.stop_condition.value
        return data
    
    async def _append_turn_to_file(self, message: Message):
        """Append a single turn to the conversation file."""
        if not self.conversation_file_path:
            return
            
        # Read current file
        with open(self.conversation_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Append new message
        data["conversation"].append(asdict(message))
        
        # Update metadata
        data["metadata"]["total_turns"] = len(self.conversation_history)
        data["metadata"]["duration_minutes"] = (time.time() - self.start_time) / 60
        
        # Write back to file
        with open(self.conversation_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _finalize_conversation_file(self):
        """Update final metadata when conversation ends."""
        if not self.conversation_file_path:
            return
            
        # Read current file
        with open(self.conversation_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update final metadata
        data["metadata"]["status"] = "completed"
        data["metadata"]["total_turns"] = len(self.conversation_history)
        data["metadata"]["duration_minutes"] = (time.time() - self.start_time) / 60
        data["metadata"]["end_timestamp"] = datetime.now().isoformat()
        
        # Write back to file
        with open(self.conversation_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation finalized: {self.conversation_file_path}")

    async def run_conversation(self) -> Dict[str, Any]:
        """Run the conversation between the two models."""
        self.start_time = time.time()
        
        # Initialize conversation file
        await self._initialize_conversation_file()
        
        # Determine who goes first (alternating)
        turn_number = 0
        current_model = self.model1 if turn_number % 2 == 0 else self.model2
        other_model = self.model2 if current_model == self.model1 else self.model1
        
        # Initialize conversation with starting question
        messages = [
            {"role": "system", "content": self._create_system_message(current_model, other_model, True)},
            {"role": "user", "content": f"Please start a conversation about: {self.config.starting_question}"}
        ]
        
        print(f"\n🤖 Starting conversation between:")
        print(f"   Model 1: {self.model1.provider.value}/{self.model1.model_name}")
        print(f"   Model 2: {self.model2.provider.value}/{self.model2.model_name}")
        print(f"   Topic: {self.config.starting_question}")
        print(f"   Stop condition: {self.config.stop_condition.value}")
        print("=" * 80)
        
        async with aiohttp.ClientSession() as session:
            while self._should_continue(turn_number):
                try:
                    # Get response from current model
                    response = await self._make_api_call(current_model, messages, session)
                    
                    # Create message object
                    message = Message(
                        role="assistant",
                        content=response,
                        timestamp=time.time(),
                        model_info={
                            "provider": current_model.provider.value,
                            "model_name": current_model.model_name
                        },
                        turn_number=turn_number
                    )
                    
                    self.conversation_history.append(message)
                    
                    # Display the message with streaming effect
                    print(f"\n🤖 {current_model.provider.value}/{current_model.model_name} (Turn {turn_number + 1}):")
                    print("-" * 50)
                    
                    # Simulate streaming by printing character by character
                    import sys
                    for char in response:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        await asyncio.sleep(0.03)  # Slower delay for visible streaming effect
                    print("\n")
                    
                    # Append this turn to the conversation file
                    if self.config.save_to_file:
                        await self._append_turn_to_file(message)
                    
                    # Add to conversation history for next model
                    messages.append({"role": "assistant", "content": response})
                    
                    # Switch models
                    turn_number += 1
                    current_model, other_model = other_model, current_model
                    
                    # Update system message for the other model
                    if turn_number == 1:  # Second model's first turn
                        messages[0] = {"role": "system", "content": self._create_system_message(current_model, other_model, False)}
                    
                    # Small pause between turns
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"\n❌ Error with {current_model.provider.value}/{current_model.model_name}: {e}")
                    break
        
        print("\n" + "=" * 80)
        print("🏁 Conversation ended")
        
        # Update final metadata
        if self.config.save_to_file:
            await self._finalize_conversation_file()
        
        def serialize_model_config(config: ModelConfig) -> Dict[str, Any]:
            data = asdict(config)
            data["provider"] = config.provider.value
            return data
        
        def serialize_conversation_config(config: ConversationConfig) -> Dict[str, Any]:
            data = asdict(config)
            data["stop_condition"] = config.stop_condition.value
            return data
        
        return {
            "conversation_history": [asdict(msg) for msg in self.conversation_history],
            "total_turns": len(self.conversation_history),
            "duration_minutes": (time.time() - self.start_time) / 60,
            "model1_config": serialize_model_config(self.model1),
            "model2_config": serialize_model_config(self.model2),
            "conversation_config": serialize_conversation_config(self.config)
        }
    
    async def _save_conversation(self):
        """Save conversation to JSON file."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{self.model1.provider.value}_{self.model2.provider.value}_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename
        
        def serialize_model_config(config: ModelConfig) -> Dict[str, Any]:
            data = asdict(config)
            data["provider"] = config.provider.value
            return data
        
        def serialize_conversation_config(config: ConversationConfig) -> Dict[str, Any]:
            data = asdict(config)
            data["stop_condition"] = config.stop_condition.value
            return data
        
        conversation_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model1": serialize_model_config(self.model1),
                "model2": serialize_model_config(self.model2),
                "config": serialize_conversation_config(self.config),
                "total_turns": len(self.conversation_history),
                "duration_minutes": (time.time() - self.start_time) / 60
            },
            "conversation": [asdict(msg) for msg in self.conversation_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        #print(f"💾 Conversation saved to: {filepath}")

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_config_from_env() -> Dict[str, str]:
    """Load API keys from environment variables."""
    return {
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "meta_api_key": os.getenv("META_API_KEY"),  # or together.ai key
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    }

async def main():
    parser = argparse.ArgumentParser(description="AI Model Conversation Tool")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--question", type=str, required=True, help="Starting question for conversation")
    parser.add_argument("--model1-provider", type=str, choices=[p.value for p in Provider], help="First model provider")
    parser.add_argument("--model1-name", type=str, help="First model name")
    parser.add_argument("--model2-provider", type=str, choices=[p.value for p in Provider], help="Second model provider")
    parser.add_argument("--model2-name", type=str, help="Second model name")
    parser.add_argument("--stop-condition", type=str, choices=[s.value for s in StopCondition], 
                       default="fixed_turns", help="Stop condition")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns (for fixed_turns)")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Time limit in minutes (for time_limit)")
    parser.add_argument("--output-dir", type=str, default="conversations", help="Output directory for logs")
    
    args = parser.parse_args()
    
    # Load environment variables
    env_config = load_config_from_env()
    
    # Load file config if provided
    file_config = {}
    if args.config:
        file_config = load_config_from_file(args.config)
    
    # Create model configurations
    if args.model1_provider and args.model1_name:
        model1 = ModelConfig(
            provider=Provider(args.model1_provider),
            model_name=args.model1_name,
            api_key=env_config.get(f"{args.model1_provider}_api_key"),
            base_url=env_config.get(f"{args.model1_provider}_base_url")
        )
    else:
        # Use config file
        m1_config = file_config["model1"]
        model1 = ModelConfig(
            provider=Provider(m1_config["provider"]),
            model_name=m1_config["model_name"],
            api_key=env_config.get(f"{m1_config['provider']}_api_key") or m1_config.get("api_key"),
            base_url=env_config.get(f"{m1_config['provider']}_base_url") or m1_config.get("base_url"),
            max_tokens=m1_config.get("max_tokens", 2000),
            temperature=m1_config.get("temperature", 0.7)
        )
    
    if args.model2_provider and args.model2_name:
        model2 = ModelConfig(
            provider=Provider(args.model2_provider),
            model_name=args.model2_name,
            api_key=env_config.get(f"{args.model2_provider}_api_key"),
            base_url=env_config.get(f"{args.model2_provider}_base_url")
        )
    else:
        # Use config file
        m2_config = file_config["model2"]
        model2 = ModelConfig(
            provider=Provider(m2_config["provider"]),
            model_name=m2_config["model_name"],
            api_key=env_config.get(f"{m2_config['provider']}_api_key") or m2_config.get("api_key"),
            base_url=env_config.get(f"{m2_config['provider']}_base_url") or m2_config.get("base_url"),
            max_tokens=m2_config.get("max_tokens", 2000),
            temperature=m2_config.get("temperature", 0.7)
        )
    
    # Create conversation configuration
    conv_config = ConversationConfig(
        stop_condition=StopCondition(args.stop_condition),
        max_turns=args.max_turns if args.stop_condition == "fixed_turns" else None,
        time_limit_minutes=args.time_limit if args.stop_condition == "time_limit" else None,
        starting_question=args.question,
        save_to_file=True,
        output_dir=args.output_dir
    )
    
    # Run conversation
    conversation = AIModelConversation(model1, model2, conv_config)
    result = await conversation.run_conversation()
    
    print(f"\n📊 Conversation Summary:")
    print(f"   Total turns: {result['total_turns']}")
    print(f"   Duration: {result['duration_minutes']:.2f} minutes")

if __name__ == "__main__":
    asyncio.run(main())
