# -*- coding: utf-8 -*-
"""
RAG Knowledge Base Q&A System - Week 3 Project
Flask backend: document ingestion, vector storage, retrieval-augmented generation
"""

import os
import uuid
import json
import hashlib
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
CHROMA_DIR = BASE_DIR / 'chroma_db'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
CORS(app)

# ============================================================
# Globals (initialized at startup)
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embedding model

# ChromaDB collection
chroma_collection = None

# Document metadata store (doc_id -> {name, chunks, created_at, size})
doc_metadata = {}
METADATA_FILE = BASE_DIR / 'doc_metadata.json'

# DeepSeek API config
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-chat'

SYSTEM_PROMPT = (
    "你是一个基于知识库的问答助手。请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请如实告知。回答时请引用来源。"
)

TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ============================================================
# Helper Functions
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(filepath):
    """SHA256 hash for file deduplication."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def parse_pdf(filepath):
    """Extract text from PDF using pdfplumber."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return '\n\n'.join(text_parts)

def parse_txt(filepath):
    """Read text file with encoding detection."""
    for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Fallback
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks using LangChain."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=['\n\n', '\n', '。', '！', '？', '.', '!', '?', '；', ';', ' ', ''],
    )
    return splitter.split_text(text)

def init_chroma():
    """Initialize ChromaDB persistent client and collection."""
    global chroma_collection
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device=str(DEVICE),
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    chroma_collection = client.get_or_create_collection(
        name='rag_documents',
        embedding_function=embedding_fn,
        metadata={'hnsw:space': 'cosine'},
    )
    print(f'ChromaDB initialized (collection: {chroma_collection.count()} chunks)')

def load_metadata():
    """Load document metadata from JSON file."""
    global doc_metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            doc_metadata = json.load(f)

def save_metadata():
    """Save document metadata to JSON file."""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

# ============================================================
# Routes: Page
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# Routes: Document Management
# ============================================================

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents with chunk counts."""
    docs = []
    for doc_id, meta in doc_metadata.items():
        docs.append({
            'id': doc_id,
            'name': meta['name'],
            'chunks': meta['chunks'],
            'size': meta['size'],
            'created_at': meta['created_at'],
        })
    docs.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'documents': docs, 'total_chunks': chroma_collection.count() if chroma_collection else 0})

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF or TXT document."""
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '仅支持PDF和TXT格式'}), 400

    # Save file with original name preserved
    original_filename = file.filename
    ext = os.path.splitext(original_filename)[1].lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'不支持的文件类型: .{ext if ext else "未知"}，仅支持PDF和TXT'}), 400

    safe_name = secure_filename(original_filename)
    # If secure_filename stripped everything (Chinese names), generate a UUID name
    if not safe_name or safe_name == ext:
        safe_name = f'{uuid.uuid4().hex[:8]}.{ext}'
    # If secure_filename kept content but lost the dot, add extension back
    if '.' not in safe_name:
        safe_name = f'{safe_name}.{ext}'

    filepath = UPLOAD_DIR / safe_name
    file.save(str(filepath))

    # Check duplicates
    file_hash = get_file_hash(filepath)
    for doc_id, meta in doc_metadata.items():
        if meta.get('hash') == file_hash:
            os.remove(filepath)
            return jsonify({'error': '该文档已存在', 'doc_id': doc_id}), 409

    # Parse text
    try:
        if ext == 'pdf':
            text = parse_pdf(filepath)
        else:
            text = parse_txt(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'文档解析失败: {str(e)}'}), 400

    if not text or len(text.strip()) < 10:
        os.remove(filepath)
        return jsonify({'error': '文档内容为空或过短'}), 400

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        os.remove(filepath)
        return jsonify({'error': '无法切分文档'}), 400

    # Generate doc_id and embed
    doc_id = str(uuid.uuid4())[:8]
    chunk_ids = [f'{doc_id}_chunk_{i}' for i in range(len(chunks))]

    chroma_collection.add(
        ids=chunk_ids,
        documents=chunks,
        metadatas=[{'doc_id': doc_id, 'doc_name': original_filename, 'chunk_idx': i} for i in range(len(chunks))],
    )

    # Store metadata
    doc_metadata[doc_id] = {
        'name': original_filename,
        'safe_name': safe_name,
        'hash': file_hash,
        'chunks': len(chunks),
        'size': os.path.getsize(filepath),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_metadata()

    return jsonify({
        'success': True,
        'doc_id': doc_id,
        'name': original_filename,
        'chunks': len(chunks),
        'total_chunks': chroma_collection.count(),
    })

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its vectors."""
    if doc_id not in doc_metadata:
        return jsonify({'error': '文档不存在'}), 404

    # Remove from ChromaDB
    chunk_ids = [f'{doc_id}_chunk_{i}' for i in range(doc_metadata[doc_id]['chunks'])]
    chroma_collection.delete(ids=chunk_ids)

    # Remove file
    filepath = UPLOAD_DIR / doc_metadata[doc_id]['safe_name']
    if filepath.exists():
        os.remove(filepath)

    # Remove metadata
    del doc_metadata[doc_id]
    save_metadata()

    return jsonify({'success': True, 'total_chunks': chroma_collection.count()})

# ============================================================
# Routes: Chat (RAG Q&A with SSE streaming)
# ============================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """RAG chat endpoint with SSE streaming."""
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求格式错误'}), 400

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': '问题不能为空'}), 400

    if chroma_collection is None or chroma_collection.count() == 0:
        return jsonify({'error': '知识库为空，请先上传文档'}), 400

    if not DEEPSEEK_API_KEY:
        return jsonify({'error': 'DeepSeek API Key 未配置，请设置环境变量 DEEPSEEK_API_KEY'}), 500

    # 1. Retrieve relevant chunks
    results = chroma_collection.query(
        query_texts=[question],
        n_results=TOP_K,
    )

    retrieved_docs = results['documents'][0] if results['documents'] else []
    retrieved_sources = results['metadatas'][0] if results['metadatas'] else []

    if not retrieved_docs:
        return jsonify({'error': '未找到相关内容'}), 404

    # 2. Build context
    context_parts = []
    sources_info = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_sources)):
        doc_name = meta.get('doc_name', 'unknown')
        context_parts.append(f'[Source {i+1} from "{doc_name}"]\n{doc}')
        if doc_name not in [s['name'] for s in sources_info]:
            sources_info.append({'name': doc_name, 'count': 1})
        else:
            for s in sources_info:
                if s['name'] == doc_name:
                    s['count'] += 1

    context = '\n\n---\n\n'.join(context_parts)

    # 3. Build prompt
    user_prompt = f'参考资料\n\n{context}\n\n用户问题: {question}\n\n回答:'

    # 4. Stream from DeepSeek
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def generate():
        full_response = ''
        try:
            stream = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f'data: {json.dumps({"type": "token", "content": content}, ensure_ascii=False)}\n\n'

            # Send sources info
            yield f'data: {json.dumps({"type": "sources", "sources": sources_info}, ensure_ascii=False)}\n\n'
            yield f'data: {json.dumps({"type": "done"}, ensure_ascii=False)}\n\n'

        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )

@app.route('/api/health', methods=['GET'])
def health():
    """Health check with system status."""
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'embedding_model': 'all-MiniLM-L6-v2',
        'documents': len(doc_metadata),
        'total_chunks': chroma_collection.count() if chroma_collection else 0,
        'llm': DEEPSEEK_MODEL if DEEPSEEK_API_KEY else '未配置',
    })

# ============================================================
# Startup
# ============================================================

if __name__ == '__main__':
    PORT = 5002
    print('=' * 60)
    print('  RAG Knowledge Base Q&A System - Week 3')
    print('=' * 60)

    # Ensure directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    load_metadata()

    # Init ChromaDB (loads embedding model internally)
    print('Initializing ChromaDB and embedding model...')
    init_chroma()

    if DEEPSEEK_API_KEY:
        print(f'DeepSeek API: configured ({DEEPSEEK_MODEL})')
    else:
        print('DeepSeek API: NOT configured - set DEEPSEEK_API_KEY env variable')
        print('  Register at: https://platform.deepseek.com')

    print(f'Documents: {len(doc_metadata)} ({chroma_collection.count()} chunks)')
    print(f'\nServer: http://127.0.0.1:{PORT}')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    app.run(host='127.0.0.1', port=PORT, debug=False)
