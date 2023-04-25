import sys
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext, download_loader
from langchain import OpenAI

import os
# Add your API keys here
os.environ["OPENAI_API_KEY"] = ""
os.environ["DISCORD_TOKEN"] = ""

# Add a discord channel ID here
channelId = 0
prompt = sys.argv[1]
print(f"prompt = {prompt}")
try:

    index = GPTSimpleVectorIndex.load_from_disk(f'{channelId}.json')
    response = index.query(prompt, similarity_top_k=3, response_mode="compact")
    print(response)
    exit
except:
    print('No existing data, fetching new (this may take a while...)')

    DiscordReader = download_loader('DiscordReader')

    discord_token = os.getenv("DISCORD_TOKEN")
    channel_ids = [channelId]  # Replace with your channel_id
    reader = DiscordReader(discord_token=discord_token)
    documents = reader.load_data(channel_ids=channel_ids)

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    index = GPTSimpleVectorIndex(nodes)
    index.save_to_disk(f'{channelId}.json')

    response = index.query(prompt, similarity_top_k=3, response_mode="compact")
    print(response)