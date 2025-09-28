import os
import json
import asyncio
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [{
                'id': f"{doc_id}_chunk_0",
                'text': text,
                'metadata': {
                    'source_doc_id': doc_id,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_break = -1
                
                for i in range(end, search_start, -1):
                    if text[i-1] in '.!?':
                        sentence_break = i
                        break
                
                if sentence_break > start:
                    end = sentence_break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': f"{doc_id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'metadata': {
                        'source_doc_id': doc_id,
                        'chunk_index': chunk_index,
                        'char_start': start,
                        'char_end': end
                    }
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                break
        
        # Update total chunks count
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks

class DataIngester:
    def __init__(self, embedding_service_url: str):
        self.embedding_service_url = embedding_service_url
        self.chunker = TextChunker(
            chunk_size=int(os.getenv('CHUNK_SIZE', 500)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50))
        )
        self.processed_hashes = set()
        self.hash_file = 'processed_docs.json'
        self.load_processed_hashes()
    
    def load_processed_hashes(self):
        """Load previously processed document hashes for idempotency"""
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r') as f:
                    data = json.load(f)
                    self.processed_hashes = set(data.get('hashes', []))
                    print(f"Loaded {len(self.processed_hashes)} processed document hashes")
            except Exception as e:
                print(f"Error loading processed hashes: {e}")
    
    def save_processed_hashes(self):
        """Save processed document hashes"""
        try:
            with open(self.hash_file, 'w') as f:
                json.dump({'hashes': list(self.processed_hashes)}, f, indent=2)
        except Exception as e:
            print(f"Error saving processed hashes: {e}")
    
    def get_document_hash(self, content: str) -> str:
        """Generate hash for document content to check if already processed"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def read_text_file(self, file_path: str) -> str:
        """Read text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def read_json_file(self, file_path: str) -> List[Dict]:
        """Read JSON file with documents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'documents' in data:
                    return data['documents']
                else:
                    return [data]
            else:
                print(f"Unsupported JSON structure in {file_path}")
                return []
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return []
    
    async def process_documents(self, input_path: str, doc_type: str = 'auto'):
        """Process documents from file or directory"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"Error: Path {input_path} does not exist")
            return
        
        documents = []
        
        if input_path.is_file():
            documents = await self.process_single_file(input_path, doc_type)
        elif input_path.is_dir():
            documents = await self.process_directory(input_path, doc_type)
        else:
            print(f"Error: {input_path} is neither a file nor directory")
            return
        
        if documents:
            await self.ingest_documents(documents)
        else:
            print("No documents to process")
    
    async def process_single_file(self, file_path: Path, doc_type: str) -> List[Dict]:
        """Process a single file"""
        print(f"Processing file: {file_path}")
        
        # Determine file type
        if doc_type == 'auto':
            if file_path.suffix.lower() == '.json':
                doc_type = 'json'
            else:
                doc_type = 'text'
        
        documents = []
        
        if doc_type == 'json':
            json_docs = self.read_json_file(str(file_path))
            for i, doc in enumerate(json_docs):
                if isinstance(doc, dict) and 'text' in doc:
                    doc_id = doc.get('id', f"{file_path.stem}_{i}")
                    documents.append({
                        'id': doc_id,
                        'text': doc['text'],
                        'metadata': {
                            'source_file': str(file_path),
                            'file_type': 'json',
                            **doc.get('metadata', {})
                        }
                    })
        else:
            content = self.read_text_file(str(file_path))
            if content:
                documents.append({
                    'id': file_path.stem,
                    'text': content,
                    'metadata': {
                        'source_file': str(file_path),
                        'file_type': 'text'
                    }
                })
        
        return documents
    
    async def process_directory(self, dir_path: Path, doc_type: str) -> List[Dict]:
        """Process all files in a directory"""
        print(f"Processing directory: {dir_path}")
        
        documents = []
        
        # Supported file extensions
        if doc_type == 'auto':
            extensions = ['.txt', '.md', '.json']
        elif doc_type == 'json':
            extensions = ['.json']
        else:
            extensions = ['.txt', '.md']
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                file_docs = await self.process_single_file(file_path, doc_type)
                documents.extend(file_docs)
        
        return documents
    
    async def ingest_documents(self, documents: List[Dict]):
        """Ingest documents into the vector store"""
        print(f"Ingesting {len(documents)} documents...")
        
        # Process documents and create chunks
        all_chunks = []
        new_documents = 0
        
        for doc in documents:
            doc_hash = self.get_document_hash(doc['text'])
            
            # Check if already processed (idempotency)
            if doc_hash in self.processed_hashes:
                print(f"Skipping already processed document: {doc['id']}")
                continue
            
            # Create chunks
            chunks = self.chunker.chunk_text(doc['text'], doc['id'])
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk['metadata'].update(doc.get('metadata', {}))
            
            all_chunks.extend(chunks)
            self.processed_hashes.add(doc_hash)
            new_documents += 1
        
        if not all_chunks:
            print("No new documents to process")
            return
        
        print(f"Created {len(all_chunks)} chunks from {new_documents} new documents")
        
        # Send to embedding service in batches
        batch_size = 10
        successful_batches = 0
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                try:
                    await self.send_batch_to_embedding_service(session, batch)
                    successful_batches += 1
                    print(f"Processed batch {successful_batches} ({len(batch)} chunks)")
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
        
        # Save processed hashes
        self.save_processed_hashes()
        
        print(f"Successfully processed {successful_batches * batch_size} chunks")
    
    async def send_batch_to_embedding_service(self, session: aiohttp.ClientSession, chunks: List[Dict]):
        """Send a batch of chunks to the embedding service"""
        payload = {
            'documents': [
                {'id': chunk['id'], 'text': chunk['text']}
                for chunk in chunks
            ]
        }
        
        async with session.post(
            f"{self.embedding_service_url}/bulk_embed",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")

async def main():
    parser = argparse.ArgumentParser(description='Ingest documents into RAG system')
    parser.add_argument('input_path', help='Path to file or directory containing documents')
    parser.add_argument('--type', choices=['text', 'json', 'auto'], default='auto',
                       help='Document type (default: auto)')
    parser.add_argument('--embedding-url', default=None,
                       help='Embedding service URL (default: from .env)')
    
    args = parser.parse_args()
    
    # Get embedding service URL
    embedding_url = args.embedding_url or os.getenv('EMBEDDING_SERVICE_URL', 'http://localhost:8000')
    
    # Initialize ingester
    ingester = DataIngester(embedding_url)
    
    # Process documents
    await ingester.process_documents(args.input_path, args.type)

if __name__ == "__main__":
    asyncio.run(main())
