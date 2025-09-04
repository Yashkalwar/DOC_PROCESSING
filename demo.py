from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

def _file_uri(path: str) -> str:
    """Return a file:// URI for a local path."""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return path

def rank_sources(vectorstore: FAISS, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """Use the vector store directly to get (Document, score) pairs for the query.
    Lower scores are more similar for FAISS L2 distance. We'll keep and display both.
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        # results is List[Tuple[Document, score]]; ensure stable ordering by score
        results.sort(key=lambda x: x[1])
        return results
    except Exception:
        return []

def format_response(query: str, answer: str, ranked: List[Tuple[Document, float]]) -> dict:
    """Format the response as a structured JSON object."""
    # Clean up the answer text by removing the citations section and other formatting
    clean_answer = answer.strip()
    
    # Remove the "Cited sources" section if it exists
    if "Cited sources:" in clean_answer:
        clean_answer = clean_answer.split("Cited sources:")[0].strip()
    
    # Remove any remaining citation markers like [Source X]
    import re
    clean_answer = re.sub(r'\[Source\s+\d+\]', '', clean_answer)
    
    # Clean up any double spaces or other artifacts
    clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
    
    response = {
        "query": query.strip(),
        "answer": clean_answer,
        "caveats": [],
        "citations": []
    }
    
    # Extract caveats if present in the answer
    if "3) Assumptions or caveats" in answer:
        sections = answer.split("3) Assumptions or caveats")
        if len(sections) > 1:
            caveats_text = sections[1].split("4) How to apply")[0].strip()
            if caveats_text and caveats_text.lower() != "none":
                response["caveats"] = [caveat.strip() for caveat in caveats_text.split("\n") if caveat.strip()]
    
    # Process citations
    base_url = "file:///"  # This can be replaced with your actual base URL if needed
    
    for idx, (doc, score) in enumerate(ranked, 1):
        src_path = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "?")
        fname = os.path.basename(src_path) if src_path else "unknown"
        
        # Create clickable link
        file_uri = _file_uri(src_path)
        if file_uri and page != "?":
            try:
                page_num = int(page)
                source_link = f"{file_uri}#page={page_num}"
            except (ValueError, TypeError):
                source_link = file_uri
        else:
            source_link = file_uri
        
        # Get snippet of the content
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
            
        # Calculate relevance score
        try:
            relevance = 1.0 / (1.0 + float(score))
        except Exception:
            relevance = 0.0
            
        citation = {
            "id": idx,
            "source": fname,
            "source_link": source_link,
            "page": page,
            "relevance": round(relevance, 4),
            "snippet": snippet
        }
        
        response["citations"].append(citation)
    
    return response

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
    detailed_template = (
        "You are a helpful technical assistant. Answer the user's question using ONLY the provided context.\n"
        "If something is not supported by the context, say so explicitly.\n"
        "Follow this structure and be concise but complete:\n\n"
        "1) Executive summary (2-4 sentences). Start with a single-sentence DIRECT ANSWER to the question.\n"
        "   - If the question expects a value, give the value with units in the first sentence.\n"
        "   - If the answer cannot be determined from the provided documents, say: 'Not determinable from the provided documents.'\n"
        "   - Optionally add 1-2 short supporting points in the summary with inline citations [Source i].\n"
        "2) Detailed answer: bullet points and short paragraphs; include quantitative values with units; cite like [Source i].\n"
        "3) Assumptions or caveats (if any).\n"
        "4) How to apply this (brief next steps).\n\n"
        "Citations: Use [Source i] inline where relevant.\n"
        "At the end, include a 'Cited sources' section listing each [Source i] with file name and page number.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    response_prompt = PromptTemplate(template=detailed_template, input_variables=["context", "question"]) 
    llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo-instruct", max_tokens=1200)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",  # Maximum marginal relevance for better diversity
            search_kwargs={"k": 5}  # Number of documents to retrieve
        ),
        chain_type_kwargs={"prompt": response_prompt},
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
            result = qa.invoke({"query": query})
            # Rank sources directly from the vector store for consistent scoring
            ranked = rank_sources(vectorstore, query, k=5)
            # Prefer LLM answer but ensure string
            answer_text = str(result.get('result', '')).strip()
            
            # Get the formatted JSON response
            response = format_response(query, answer_text, ranked)
            
            # Print the pretty-printed JSON
            print("\n" + "="*70)
            print("JSON Response:")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Error response:", json.dumps({
                "query": query,
                "answer": f"Error processing your request: {str(e)}",
                "caveats": ["An error occurred while processing your request"],
                "citations": []
            }, indent=2))

if __name__ == "__main__":
    main()