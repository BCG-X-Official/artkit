import asyncio
import json
import artkit.api as ak

async def test_titan_model():
    try:
        # Initialize a Titan chat model
        titan_bedrock_model = ak.CachedChatModel(
            model=ak.TitanBedrockChat(
                model_id='amazon.titan-text-lite-v1',
                # model_id = 'amazon.titan-embed-text-v2:0',
                region='us-east-1'),
            database="cache/connecting_to_titan1.db",
        )

        # Send a message to the chat model and get the response
        response = await titan_bedrock_model.get_response(
            json.dumps({"inputText": "What is the best Javascript framework?"})
        )
        print("Full Response:", response)
        # print(response[0])
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test
asyncio.run(test_titan_model())


