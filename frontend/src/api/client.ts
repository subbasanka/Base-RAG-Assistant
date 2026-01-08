import axios from 'axios';
import type { QueryRequest, QueryResponse, HealthResponse, UploadResponse, ConfigResponse } from '../types/api';

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
