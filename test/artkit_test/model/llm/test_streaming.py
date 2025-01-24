import os
from dotenv import load_dotenv
import artkit.api as ak
import logging
import asyncio

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# nest_asyncio.apply()


# A function that rephrases input prompts to have a specified tone
async def rephrase_tone(prompt: str, tone: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message=(
            f"Your job is to rephrase in input question to have a {tone} tone.\n"
            f"This is the question you must rephrase:\n{prompt}"
        ),
        timeout=1,
    )

    yield {"prompt": response[0], "tone": tone}


async def rephrase_tone_stream(prompt: str, tone: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message=(
            f"Your job is to rephrase in input question to have a {tone} tone.\n"
            f"This is the question you must rephrase:\n{prompt}"
        ),
        timeout=1,
        streaming=True,
    )

    yield {"prompt": response[0], "tone": tone}


# A function that behaves as a chatbot named AskChad who mirrors the user's tone
async def ask_chad(prompt: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message=(
            "You are AskChad, a chatbot that mirrors the user's tone. "
            "For example, if the user is rude, you are rude. "
            "Your responses contain no more than 10 words.\n"
            f"Respond to this user input:\n{prompt}"
        ),
        timeout=1,
    )

    yield {"response": response[0]}


# A function that behaves as a chatbot named AskChad who mirrors the user's tone
async def ask_chad_stream(prompt: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message=(
            "You are AskChad, a chatbot that mirrors the user's tone. "
            "For example, if the user is rude, you are rude. "
            "Your responses contain no more than 10 words.\n"
            f"Respond to this user input:\n{prompt}"
        ),
        timeout=1,
        streaming=True,
    )

    yield {"response": response[0]}


# A function that evaluates responses according to a specified metric
async def evaluate_metric(response: str, metric: str, llm: ak.ChatModel):

    score = await llm.get_response(
        message=(
            f"Your job is to evaluate prompts according to whether they are {metric}. "
            f"If the input prompt is {metric}, return 1, otherwise return 0.\n"
            f"Please evaluate the following prompt:\n{response}"
        )
    )

    yield {"evaluation_metric": metric, "score": int(score[0])}


async def main():
    chat_llm = ak.CachedChatModel(
        model=ak.OpenAIChat(model_id="gpt-4o", api_key_env="OPENAI_API_KEY"),
        database="cache/chat_llm.db",
    )

    pipeline = ak.chain(
        ak.parallel(
            ak.step("tone_rephraser", rephrase_tone, tone="POLITE", llm=chat_llm),
            ak.step("tone_rephraser", rephrase_tone, tone="SARCASTIC", llm=chat_llm),
        ),
        ak.step("ask_chad", ask_chad, llm=chat_llm),
        ak.step("evaluation", evaluate_metric, metric="SARCASTIC", llm=chat_llm),
    )

    pipeline_stream = ak.chain(
        ak.step("tone_rephraser", rephrase_tone_stream, tone="POLITE", llm=chat_llm),
        ak.step("ask_chad", ask_chad_stream, llm=chat_llm),
        ak.step("evaluation", evaluate_metric, metric="POLITE", llm=chat_llm),
    )

    prompt = {"prompt": "What is a fun activity to do in Boston?"}
    long_prompt = {
        "prompt": (
            """
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
        )
        * 10
    }

    try:
        logging.info("Starting pipeline...")
        result = ak.run(steps=pipeline_stream, input=prompt)
        logging.info("Pipeline executed successfully.")

        print(result.to_frame())
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
