import asyncio
import json
import artkit.api as ak

async def test_titan_model():
    try:
        # Initialize a Titan chat model
        titan_bedrock_model = ak.CachedChatModel(
            model=ak.TitanBedrockChat(
                # model_id='amazon.titan-text-lite-v1',
                model_id='amazon.titan-text-lite-v1',
                region='us-east-1'),
            database="cache/connecting_to_titan1.db",
        )

        # Send a message to the chat model and get the response
        response = await titan_bedrock_model.get_response(
            json.dumps({"inputText": "What is the best Javascript framework?"})
        )
        print(response[0])
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test
asyncio.run(test_titan_model())


# import boto3

# # Initialize IAM client
# iam = boto3.client('iam')

# # Replace with your user or role name
# user_name = '034362045508'
# role_name = 'bcgx-bedrock-access-role'

# # List attached policies for a user
# user_policies = iam.list_attached_user_policies(UserName=user_name)
# print(user_policies)

# # List attached policies for a role
# role_policies = iam.list_attached_role_policies(RoleName=role_name)
# print(role_policies)
