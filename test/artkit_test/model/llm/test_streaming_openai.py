import asyncio
import os

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Long prompt for testing
long_prompt = """
You are an AI assistant tasked with generating an extensive and multifaceted document on the following topics:
1. **Interplanetary Political Systems**: A 3000-word analysis of the political structures governing a federation of 50 planets, including:
   - Detailed descriptions of each planet's unique political system.
   - The process of electing interplanetary representatives and resolving conflicts.
   - Case studies of three hypothetical political crises and their resolutions.
2. **Interstellar Economy**: A 3000-word detailed breakdown of:
   - The trade networks, currency systems, and taxation policies across planets.
   - The effects of hyperinflation and interstellar market crashes, with proposed recovery plans.
   - A mathematical model (with equations) illustrating the balance of trade and resource allocation.
3. **Cultural Exchange**: A 3000-word discussion of:
   - Linguistic, artistic, and religious exchanges among the planets.
   - The role of interplanetary festivals and their impact on diplomacy.
   - Detailed character-driven stories illustrating cultural misunderstandings and resolutions.
4. **Technological Innovations**: A 3000-word technical overview of:
   - Space travel advancements, including propulsion systems, quantum communication, and AI-assisted navigation.
   - The evolution of terraforming technologies, with case studies of three planets.
   - A technical schematic of a futuristic space station, complete with annotations for key modules.
5. **Creative Writing Task**: Write a 5000-word science fiction story:
   - The story should feature five well-developed characters from different planets.
   - Include a high-stakes interstellar diplomatic mission that goes awry.
   - Use rich, vivid descriptions of planets, space stations, and alien ecosystems.
   - Integrate political intrigue, action sequences, and philosophical debates about humanity's future.

Ensure each section is written with extreme detail, integrating plausible science, hypothetical scenarios, and deep narrative elements. Connect the sections cohesively so that they reflect a single, unified universe.
"""


async def get_response_without_streaming():
    try:
        async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": (long_prompt) * 10},
            ],
            timeout=1,
        )
        print("Response received (non-streaming):")
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error without streaming: {e}")


# Function for streaming response
async def get_response_with_streaming():
    try:
        async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": (long_prompt) * 10},
            ],
            stream=True,  # Enable streaming
            timeout=1,
        )
        combined_content = ""
        print("Streaming response:")
        async for chunk in response:
            # Extract the content from the streamed chunk
            content = chunk.choices[0].delta.content
            if content:
                combined_content += content
            print(content, end="", flush=True)
    except Exception as e:
        print(f"Error with streaming: {e}")


# Main function to test both methods
async def main():
    print("\nWithout streaming:")
    await get_response_without_streaming()

    print("\n\nWith streaming:")
    await get_response_with_streaming()


# Run the main function
asyncio.run(main())
