# ModelTalk

<pre>
 "The nexarion, imbued with the aethonix, journeyed through the celyddon, her path illuminated by the sonderlux and the duskyn light, she frothingaled with joy, her heart filled with skylarcity, as she resonated with the floragabble and the whispers of the aerthys, her blustranaut spirit flourishing amidst the fluxion, as she discovered the gravilume, the lumifizz, and the glimmerwings in the elyrian landscapes, her eyes quinkling with creativity and innovation."
</pre>

A configurable tool for orchestrating conversations between different AI models from various providers (Anthropic, OpenAI, Google, Meta, Ollama). Watch AI models engage in real-time streaming conversations while automatically saving complete transcripts.

## Features

- **Multi-Provider Support**: Anthropic Claude, OpenAI GPT, Google Gemini, Meta Llama, and local Ollama models
- **Real-Time Streaming**: Character-by-character streaming output with visual typing effect
- **Live JSON Logging**: Conversations saved incrementally after each turn (no data loss if interrupted)
- **Flexible Stop Conditions**: Fixed turns, time limits, or let models decide when to end
- **Conversation Reader**: Clean markdown display of saved conversations
- **Configurable**: JSON config files or command-line arguments

## Quick Start

### 1. Install Dependencies
```bash
pip install aiohttp
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
# Optional: GOOGLE_API_KEY, META_API_KEY
```

### 3. Run a Conversation
```bash
# Using command line arguments
python ai_conversation.py \
  --question "Do AI models have consciousness?" \
  --model1-provider openai --model1-name gpt-4 \
  --model2-provider anthropic --model2-name claude-3-opus-20240229 \
  --max-turns 10

# Using config file
python ai_conversation.py --question "Your topic here" --config config.json
```

### 4. View Saved Conversations
```bash
# List all conversations
python conversation_reader.py --list

# Display a specific conversation
python conversation_reader.py conversations/conversation_openai_anthropic_20250531_123456.json
```

## Configuration

### Command Line Options
- `--question`: Starting topic for the conversation
- `--model1-provider/--model1-name`: First model configuration
- `--model2-provider/--model2-name`: Second model configuration
- `--stop-condition`: `fixed_turns`, `time_limit`, or `model_decides`
- `--max-turns`: Number of turns (for `fixed_turns`)
- `--time-limit`: Minutes (for `time_limit`)
- `--output-dir`: Directory for saving conversations

### Config File Format
```json
{
  "model1": {
    "provider": "openai",
    "model_name": "gpt-4",
    "max_tokens": 2000,
    "temperature": 0.7
  },
  "model2": {
    "provider": "anthropic", 
    "model_name": "claude-3-opus-20240229",
    "max_tokens": 2000,
    "temperature": 0.7
  }
}
```

## Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4, gpt-3.5-turbo, etc. | Requires `OPENAI_API_KEY` |
| **Anthropic** | claude-3-opus, claude-3-sonnet, etc. | Requires `ANTHROPIC_API_KEY` |
| **Google** | gemini-pro, gemini-1.5-pro, etc. | Requires `GOOGLE_API_KEY` |
| **Meta** | llama-2, llama-3, etc. | Via together.ai (requires `META_API_KEY`) |
| **Ollama** | Any local model | Requires local Ollama server |

## Example Usage

### Debate Between Models
```python
# test.py example
model1 = ModelConfig(
    provider=Provider.OPENAI,
    model_name="gpt-4",
    api_key="your-key"
)

model2 = ModelConfig(
    provider=Provider.OLLAMA,
    model_name="llama4:latest",
    base_url="http://localhost:11434"
)

config = ConversationConfig(
    stop_condition=StopCondition.FIXED_TURNS,
    max_turns=10,
    starting_question="Debate the best approach to AI safety",
    save_to_file=True,
    output_dir="debates"
)
```

### Creative Collaboration
```bash
python ai_conversation.py \
  --question "Create a science fiction story together, taking turns" \
  --model1-provider anthropic --model1-name claude-3-opus-20240229 \
  --model2-provider openai --model2-name gpt-4 \
  --stop-condition model_decides
```

## Output Format

Conversations are saved as JSON with complete metadata:
```json
{
  "metadata": {
    "timestamp": "2025-05-31T10:30:00Z",
    "model1": {"provider": "openai", "model_name": "gpt-4"},
    "model2": {"provider": "anthropic", "model_name": "claude-3-opus"},
    "total_turns": 10,
    "duration_minutes": 5.2,
    "status": "completed"
  },
  "conversation": [
    {
      "role": "assistant",
      "content": "Hello, I'm GPT-4...",
      "timestamp": 1234567890,
      "model_info": {"provider": "openai", "model_name": "gpt-4"},
      "turn_number": 0
    }
  ]
}
```

## Files

- `ai_conversation.py` - Main conversation orchestrator
- `conversation_reader.py` - View and format saved conversations  
- `test.py` - Example conversation setup
- `config.json` - Sample configuration file
- `CLAUDE.md` - Development instructions for Claude Code

## Use Cases

- **AI Research**: Study how different models interact and respond to each other
- **Creative Writing**: Collaborative storytelling between AI models
- **Debates**: Structured arguments between models with different perspectives
- **Problem Solving**: Multi-model approaches to complex questions
- **Model Comparison**: Observe differences in reasoning and communication styles

## Requirements

- Python 3.7+
- aiohttp
- API keys for desired providers
- Local Ollama installation (optional)
