import asyncio
from ai_conversation import AIModelConversation, ModelConfig, ConversationConfig, Provider, StopCondition

api_key="<your-openai-api-key-here"
async def run_research_conversation():
    # Configure models
    model1 = ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        api_key=api_key
    )
    
    model2 = ModelConfig(
        provider=Provider.OLLAMA,
        model_name="llama4:latest",
        base_url="http://localhost:11434"
    )
    
    # Configure conversation
    config = ConversationConfig(
        stop_condition=StopCondition.FIXED_TURNS,
        max_turns=17,
        starting_question="With each turn you will make up a word that doesn't exist and define it.  Then  you will pass that word along with all the other words the two of you invent.  You will create a sentence using all the new words. Finally you will produce a list of all the new words and their definitions with each turn.",
        save_to_file=True,
        output_dir="research_logs"
    )
    
    # Run conversation
    conversation = AIModelConversation(model1, model2, config)
    result = await conversation.run_conversation()
    
    print(f"Conversation completed with {result['total_turns']} turns")
    return result

# Run it
asyncio.run(run_research_conversation())

