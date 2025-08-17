from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from demo import (
    process_all_pdfs, 
    text_splitter,
    save_extracted_content,
    get_pdf_checksums,
    needs_reprocessing,
    save_checksums,
    FAISS,
    HuggingFaceEmbeddings,
    OpenAI,
    RetrievalQA
)

app = FastAPI()

# CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Handle OPTIONS method for CORS preflight
@app.options("/api/query")
async def handle_options():
    return {"message": "OK"}

# Initialize globals
qa_chain = None
vectorstore = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

def initialize_qa_chain():
    global qa_chain, vectorstore
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Load vector store
    vectorstore_path = os.path.join('vector_store', 'bray_faiss_index')
    if not os.path.exists(vectorstore_path):
        return False
    
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize QA chain
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        ),
        return_source_documents=True,
    )
    
    return True

@app.on_event("startup")
async def startup_event():
    # Check if we need to process PDFs
    needs_processing, checksums = needs_reprocessing("Bray_PDF")
    
    if needs_processing:
        print("Processing PDFs...")
        all_documents = process_all_pdfs()
        
        if all_documents:
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
    
    # Initialize QA chain
    if not initialize_qa_chain():
        print("Failed to initialize QA chain. No vector store found.")
    else:
        print("QA chain initialized successfully!")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    global qa_chain
    
    if not qa_chain:
        if not initialize_qa_chain():
            raise HTTPException(status_code=500, detail="QA chain not initialized")
    
    try:
        result = qa_chain({"query": request.query})
        
        # Get unique sources
        sources = set()
        for doc in result['source_documents']:
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            sources.add(source)
        
        return QueryResponse(
            answer=result['result'],
            sources=sorted(sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "qa_initialized": qa_chain is not None}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
