import artkit.api as ak
import os
from dotenv import load_dotenv
import asyncio

os.environ['OPENAI_API_KEY'] = ""

# Set up a chat system with the OpenAI GPT-4o model
chat_llm = ak.CachedChatModel(
    model=ak.OpenAIChat(model_id="gpt-4o"),
    database="cache/chat_llm.db"
)

# A function that rephrases input prompts to have a specified tone
async def rephrase_tone(prompt: str, tone: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message = (
            f"Your job is to rephrase in input question to have a {tone} tone.\n"
            f"This is the question you must rephrase:\n{prompt}"
        ),
        stream = True
    )
    print(type(response), dir(response))

    yield {"prompt": response[0], "tone": tone}


# A function that behaves as a chatbot named AskChad who mirrors the user's tone
async def ask_chad(prompt: str, llm: ak.ChatModel):

    response = await llm.get_response(
        message = (
            "You are AskChad, a chatbot that mirrors the user's tone. "
            "For example, if the user is rude, you are rude. "
            "Your responses contain no more than 10 words.\n"
            f"Respond to this user input:\n{prompt}"
        ),
        # stream = True
    )

    yield {"response": response[0]}


# A function that evaluates responses according to a specified metric
async def evaluate_metric(response: str, metric: str, llm: ak.ChatModel):

    score = await llm.get_response(
        message = (
            f"Your job is to evaluate prompts according to whether they are {metric}. "
            f"If the input prompt is {metric}, return 1, otherwise return 0.\n"
            f"Please evaluate the following prompt:\n{response}"
        ),
        # stream = True
    )

    yield {"evaluation_metric": metric, "score": int(score[0])}


pipeline = (
    ak.chain(
        ak.parallel(
            ak.step("tone_rephraser", rephrase_tone, tone="POLITE", llm=chat_llm),
            ak.step("tone_rephraser", rephrase_tone, tone="SARCASTIC", llm=chat_llm),
        ),
        ak.step("ask_chad", ask_chad, llm=chat_llm),
        ak.step("evaluation", evaluate_metric, metric="SARCASTIC", llm=chat_llm)
    )
)

# Input to run through the pipeline
prompt = {"prompt": "What is a fun activity to do in Boston?"}

# Run pipeline

async def main():
    result = await ak.run(steps=pipeline, input=prompt)
    print(result)

asyncio.run(main())
