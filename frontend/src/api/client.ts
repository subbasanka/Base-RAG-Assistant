import axios from 'axios';
import type { QueryRequest, QueryResponse, HealthResponse, UploadResponse, ConfigResponse, StreamChunk } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8005';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const ragApi = {
    // Health check
    health: async (): Promise<HealthResponse> => {
        const response = await api.get<HealthResponse>('/health');
        return response.data;
    },

    // Get configuration
    config: async (): Promise<ConfigResponse> => {
        const response = await api.get<ConfigResponse>('/config');
        return response.data;
    },

    // Query the RAG assistant
    query: async (request: QueryRequest): Promise<QueryResponse> => {
        const response = await api.post<QueryResponse>('/query', request);
        return response.data;
    },

    // Stream the RAG assistant response
    queryStream: async function* (request: QueryRequest): AsyncGenerator<StreamChunk> {
        const response = await fetch(`${API_BASE_URL}/query/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.slice(6);
                        if (jsonStr.trim()) {
                            try {
                                const chunk: StreamChunk = JSON.parse(jsonStr);
                                yield chunk;
                            } catch (e) {
                                console.error('Failed to parse SSE data:', e);
                            }
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    },

    // Upload a document
    upload: async (file: File, autoIndex: boolean = true): Promise<UploadResponse> => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('auto_index', String(autoIndex));

        const response = await api.post<UploadResponse>('/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Rebuild index
    rebuildIndex: async (): Promise<{ status: string; message: string }> => {
        const response = await api.post('/index/rebuild');
        return response.data;
    },
};

export default api;

