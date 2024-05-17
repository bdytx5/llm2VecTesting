import json
from tqdm import tqdm
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

# Load the subset of the wikimedia/wikisource dataset
dataset = load_dataset("wikimedia/wikisource", "20231201.en")

# Select the train split and limit to the first 100 examples
train_dataset = dataset['train'].select(range(1000))

# Convert dataset to LangchainDocument format
langchain_docs = [
    LangchainDocument(page_content=doc["text"], metadata={"id": doc["id"], "url": doc["url"], "title": doc["title"]})
    for doc in tqdm(train_dataset)
]

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Process documents and split into chunks
docs_processed = []
for doc in langchain_docs:
    chunks = text_splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        chunk.metadata["custom_id"] = f"{chunk.metadata['id']}_chunk_{i}"
        docs_processed.append(chunk)

# Define function to convert processed documents to JSON format
def documents_to_json(docs):
    json_list = []
    for doc in docs:
        json_item = {
            "custom_id": doc.metadata["custom_id"],
            "id": doc.metadata["id"],
            "url": doc.metadata["url"],
            "title": doc.metadata["title"],
            "text": doc.page_content
        }
        json_list.append(json_item)
    return json_list

# Convert the processed documents to JSON
json_data = documents_to_json(docs_processed)

# Save the JSON data to a file
output_file_path = "./chunked_wikisource_data.json"
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)

print(f"Chunked data has been saved to {output_file_path}")
