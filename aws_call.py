
"""
Shows how to create a list of action items from a meeting transcript
with the Amazon Titan Text model (on demand).
"""
import json
import logging
import boto3

from botocore.exceptions import ClientError


class ImageError(Exception):
    "Custom exception for errors returned by Amazon Titan Text models"

    def __init__(self, message):
        self.message = message


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_text(model_id, body):
    """
    Generate text using Amazon Titan Text models on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (json): The response from the model.
    """

    logger.info(
        "Generating text with Amazon Titan Text model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime')

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise ImageError(f"Text generation error. Error is {finish_reason}")

    logger.info(
        "Successfully generated text with Amazon Titan Text model %s", model_id)

    return response_body


def main():
    """
    Entrypoint for Amazon Titan Text model example.
    """
    try:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")

        # You can replace the model_id with any other Titan Text Models
        # Titan Text Model family model_id is as mentioned below:
        # amazon.titan-text-premier-v1:0, amazon.titan-text-express-v1, amazon.titan-text-lite-v1
        model_id = 'amazon.titan-text-lite-v1'

        prompt = """Meeting transcript: Miguel: Hi Brant, I want to discuss the workstream  
            for our new product launch Brant: Sure Miguel, is there anything in particular you want
            to discuss? Miguel: Yes, I want to talk about how users enter into the product.
            Brant: Ok, in that case let me add in Namita. Namita: Hey everyone 
            Brant: Hi Namita, Miguel wants to discuss how users enter into the product.
            Miguel: its too complicated and we should remove friction.  
            for example, why do I need to fill out additional forms?  
            I also find it difficult to find where to access the product
            when I first land on the landing page. Brant: I would also add that
            I think there are too many steps. Namita: Ok, I can work on the
            landing page to make the product more discoverable but brant
            can you work on the additonal forms? Brant: Yes but I would need 
            to work with James from another team as he needs to unblock the sign up workflow.
            Miguel can you document any other concerns so that I can discuss with James only once?
            Miguel: Sure.
            From the meeting transcript above, Create a list of action items for each person. """

        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 3072,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        })

        response_body = generate_text(model_id, body)
        print(f"Input token count: {response_body['inputTextTokenCount']}")

        for result in response_body['results']:
            print(f"Token count: {result['tokenCount']}")
            print(f"Output text: {result['outputText']}")
            print(f"Completion reason: {result['completionReason']}")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))
    except ImageError as err:
        logger.error(err.message)
        print(err.message)

    else:
        print(
            f"Finished generating text with the Amazon Titan Text Premier model {model_id}.")


if __name__ == "__main__":
    main()