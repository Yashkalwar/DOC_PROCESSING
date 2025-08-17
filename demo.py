from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import json
import glob
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path

# Create directories for extracted content
os.makedirs('extracted_images', exist_ok=True)
os.makedirs('vector_store', exist_ok=True)

load_dotenv()

# Initialize text splitter with optimized settings for technical documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Slightly smaller chunks for better context
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # Better handling of technical documents
    is_separator_regex=False,
)

def process_pdf(pdf_path: str) -> List[Document]:
    """Process a single PDF file and return a list of Document objects."""
    documents = []
    pdf_name = os.path.basename(pdf_path)
    
    try:
        # Extract tables with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    table_str = "\n".join([
                        " | ".join([str(cell).strip() if cell is not None else "" for cell in row])
                        for row in table if any(cell is not None for cell in row)
                    ])
                    if table_str.strip():
                        documents.append(Document(
                            page_content=f"[Table from {pdf_name}, page {page_num + 1}]\n{table_str}",
                            metadata={
                                "source": pdf_path,
                                "type": "table",
                                "page": page_num + 1,
                                "document": pdf_name
                            }
                        ))
        
        # Extract text and images with PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Get regular text with better formatting
            text = page.get_text("text")
            
            # Extract images
            images = page.get_images(full=True)
            image_paths = []
            
            for img_index, img in enumerate(images, 1):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Create unique image filename
                    img_filename = os.path.join(
                        'extracted_images',
                        f"{Path(pdf_name).stem}_p{page_num+1}_i{img_index}.{base_image['ext']}"
                    )
                    
                    # Save the image
                    with open(img_filename, "wb") as img_file:
                        img_file.write(base_image["image"])
                    
                    image_paths.append(img_filename)
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num + 1}: {str(e)}")
            
            # Add document with metadata
            if text.strip() or image_paths:
                doc_metadata = {
                    "source": pdf_path,
                    "page": page_num + 1,
                    "images": len(images),
                    "document": pdf_name,
                    "content_type": "text"
                }
                
                # Add image references to text
                if image_paths:
                    text += "\n\n[Images on this page: " + ", ".join(image_paths) + "]"
                
                documents.append(Document(
                    page_content=text,
                    metadata=doc_metadata
                ))
        
        return documents
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []

def process_all_pdfs(pdf_dir: str = "Bray_PDF") -> List[Document]:
    """Process all PDFs in the specified directory."""
    all_documents = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return all_documents
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {os.path.basename(pdf_file)}")
        documents = process_pdf(pdf_file)
        all_documents.extend(documents)
        print(f"  - Extracted {len(documents)} chunks")
    
    return all_documents

def save_extracted_content(documents: List[Document], output_file: str = 'extracted_content.json') -> None:
    """Save extracted content to a JSON file in the specified format."""
    output = {
        "extraction_timestamp": datetime.now().isoformat(),
        "documents": {}
    }
    
    # Group documents by source file
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        doc_name = os.path.basename(source)
        
        if doc_name not in output["documents"]:
            output["documents"][doc_name] = {
                "source": source,
                "pages": {}
            }
        
        page_num = doc.metadata.get('page', 1)
        if page_num not in output["documents"][doc_name]["pages"]:
            output["documents"][doc_name]["pages"][page_num] = {
                "page_number": page_num,
                "content": "",
                "tables": [],
                "images": []
            }
        
        page_data = output["documents"][doc_name]["pages"][page_num]
        
        if doc.metadata.get('type') == 'table':
            table_data = [row.split('|') for row in doc.page_content.split('\n') if row]
            page_data["tables"].extend(table_data)
        else:
            if doc.metadata.get('images', 0) > 0:
                for img_num in range(1, doc.metadata['images'] + 1):
                    img_id = f"i{len(page_data['images']) + 1}"
                    img_path = os.path.join('extracted_images', 
                                         f"{Path(doc_name).stem}_p{page_num}_i{img_num}.png")
                    page_data["images"].append({
                        "image_id": img_id,
                        "path": img_path,
                        "caption": f"Image from {doc_name}, page {page_num}"
                    })
            
            if doc.page_content.strip():
                if page_data["content"]:
                    page_data["content"] += "\n\n" + doc.page_content
                else:
                    page_data["content"] = doc.page_content
    
    # Convert page dictionaries to lists
    for doc_name, doc_data in output["documents"].items():
        doc_data["pages"] = [
            page_data for _, page_data in sorted(doc_data["pages"].items())
        ]
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted content saved to {output_file}")

def get_pdf_checksums(pdf_dir: str) -> Dict[str, float]:
    """Get modification times of all PDFs in the directory."""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    return {pdf: os.path.getmtime(pdf) for pdf in pdf_files}

def needs_reprocessing(pdf_dir: str, checksum_file: str = 'pdf_checksums.json') -> Tuple[bool, Dict[str, float]]:
    """Check if PDFs have been modified since last processing."""
    current_checksums = get_pdf_checksums(pdf_dir)
    
    if not os.path.exists(checksum_file) or not os.path.exists(os.path.join('vector_store', 'bray_faiss_index')):
        return True, current_checksums
    
    try:
        with open(checksum_file, 'r') as f:
            saved_checksums = json.load(f)
            
        # Check if any PDFs were added, removed, or modified
        if set(current_checksums.keys()) != set(saved_checksums.keys()):
            return True, current_checksums
            
        # Check modification times
        for pdf, mtime in current_checksums.items():
            if abs(mtime - saved_checksums.get(pdf, 0)) > 1:  # 1 second tolerance
                return True, current_checksums
                
        return False, current_checksums
    except Exception:
        return True, current_checksums

def save_checksums(checksums: Dict[str, float], checksum_file: str = 'pdf_checksums.json') -> None:
    """Save the current PDF checksums to a file."""
    with open(checksum_file, 'w') as f:
        json.dump(checksums, f)

def main():
    # Check if we need to process PDFs
    needs_processing, checksums = needs_reprocessing("Bray_PDF")
    
    if needs_processing:
        print("Processing PDFs...")
        all_documents = process_all_pdfs()
        
        if not all_documents:
            print("No documents were processed. Using existing vector store if available.")
        else:
            print(f"\nTotal documents processed: {len(all_documents)}")
            save_extracted_content(all_documents)
            
            # Create chunks from all documents
            chunks = text_splitter.split_documents(all_documents)
            print(f"Total chunks created: {len(chunks)}")
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            
            # Create or update vector store
            vectorstore_path = os.path.join('vector_store', 'bray_faiss_index')
            
            if os.path.exists(vectorstore_path):
                print("\nUpdating existing vector store...")
                vectorstore = FAISS.load_local(
                    vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                vectorstore.add_documents(chunks)
            else:
                print("\nCreating new vector store...")
                vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Save the updated vector store and checksums
            vectorstore.save_local(vectorstore_path)
            save_checksums(checksums)
            print(f"Vector store saved to {vectorstore_path}")
    
    # Load the existing vector store
    vectorstore_path = os.path.join('vector_store', 'bray_faiss_index')
    if not os.path.exists(vectorstore_path):
        print("No vector store found and no documents processed. Exiting...")
        return
        
    print("\nLoading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize QA chain
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",  # Maximum marginal relevance for better diversity
            search_kwargs={"k": 5}  # Number of documents to retrieve
        ),
        return_source_documents=True,
        verbose=True
    )
    
    # Example query
    while True:
        print("\n" + "="*50)
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            result = qa({"query": query})
            print(f"\nAnswer: {result['result']}")
            
            # Show unique PDF sources
            sources = set()
            for doc in result['source_documents']:
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                sources.add(source)
            
            if sources:
                print("\nSource PDFs:")
                for i, source in enumerate(sorted(sources), 1):
                    print(f"{i}. {source}")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()