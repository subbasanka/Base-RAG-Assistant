// API types matching FastAPI backend

export interface Source {
    content: string;
    source: string;
    page: number;
    score: number;
    citation: string;
}

export interface QueryRequest {
    query: string;
    top_k?: number;
}

export interface QueryResponse {
    answer: string;
    query: string;
    sources: Source[];
    latency_ms: number;
}

export interface HealthResponse {
    status: string;
    timestamp: number;
    model: string;
    embeddings: string;
    index_loaded: boolean;
}

export interface UploadResponse {
    status: string;
    filename: string;
    message: string;
    index_triggered: boolean;
}

export interface ConfigResponse {
    llm: {
        model_name: string;
        temperature: number;
        max_tokens: number;
    };
    retrieval: {
        top_k: number;
    };
    embeddings: {
        provider: string;
        model: string;
    };
    environment: string;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: Source[];
    latency_ms?: number;
    timestamp: Date;
}

export interface Document {
    name: string;
    size: number;
    type: string;
}
