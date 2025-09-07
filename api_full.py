from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import os
import json
from demo import (
    process_pdf,
    process_all_pdfs,
    save_extracted_content,
    get_pdf_checksums,
    needs_reprocessing,
    save_checksums,
    rank_sources,
    format_response,
    text_splitter
)
from langchain_community.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    id: int
    source: str
    source_link: str
    page: Any
    relevance: float
    snippet: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    caveats: List[str]
    citations: List[Citation]

class PDFProcessResponse(BaseModel):
    processed: bool
    total_documents: int
    total_chunks: int
    vectorstore_path: str

vectorstore = None
qa = None
vectorstore_path = os.path.join('vector_store', 'bray_faiss_index')

@app.on_event("startup")
def load_model():
    global vectorstore, qa
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
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
                search_type="mmr",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": response_prompt},
            return_source_documents=True,
            verbose=False
        )
    else:
        vectorstore = None
        qa = None

@app.post("/process_pdfs", response_model=PDFProcessResponse)
async def process_pdfs():
    needs_processing, checksums = needs_reprocessing("Bray_PDF")
    if not needs_processing:
        return PDFProcessResponse(processed=False, total_documents=0, total_chunks=0, vectorstore_path=vectorstore_path)
    all_documents = process_all_pdfs()
    save_extracted_content(all_documents)
    chunks = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vectorstore_path)
    save_checksums(checksums)
    return PDFProcessResponse(
        processed=True,
        total_documents=len(all_documents),
        total_chunks=len(chunks),
        vectorstore_path=vectorstore_path
    )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global vectorstore, qa
    if not vectorstore or not qa:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        result = qa.invoke({"query": request.query})
        ranked = rank_sources(vectorstore, request.query, k=5)
        answer_text = str(result.get('result', '')).strip()
        response = format_response(request.query, answer_text, ranked)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vectorstore_exists": os.path.exists(vectorstore_path)}

@app.get("/pdf_checksums")
async def pdf_checksums():
    checksums = get_pdf_checksums("Bray_PDF")
    return checksums

@app.post("/process_single_pdf")
async def process_single_pdf(pdf_path: str):
    documents = process_pdf(pdf_path)
    return {"pdf": pdf_path, "chunks": len(documents)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("api_full:app", host="0.0.0.0", port=8000)
