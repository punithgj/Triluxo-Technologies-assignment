import os

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

def extract_data_from_url(url):
    headers = {
        'User-Agent': os.getenv('USER_AGENT')
    }
    loader = WebBaseLoader(url, requests_kwargs={'headers': headers})
    documents = loader.load()
    return documents

def clean_data(documents):
    cleaned_documents = []
    for doc in documents:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        # Extract and clean the text
        text = soup.get_text(separator=" ", strip=True)
        cleaned_documents.append(text)
    return cleaned_documents

if __name__ == "__main__":
    url = 'https://brainlox.com/courses/category/technical'
    documents = extract_data_from_url(url)
    cleaned_documents = clean_data(documents)
    
    # Print the extracted and cleaned data
    for i, text in enumerate(cleaned_documents):
        print(f"Document {i+1}:")
        print(text)
        print("\n" + "="*80 + "\n")
