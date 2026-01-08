import { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ragApi } from '../api/client';
import type { Message, Source, QueryResponse } from '../types/api';

// Generate unique IDs
const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

/**
 * Hook for managing chat messages and querying the RAG assistant
 */
export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const queryClient = useQueryClient();

    const queryMutation = useMutation({
        mutationFn: ragApi.query,
        onSuccess: (data: QueryResponse) => {
            const assistantMessage: Message = {
                id: generateId(),
                role: 'assistant',
                content: data.answer,
                sources: data.sources,
                latency_ms: data.latency_ms,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, assistantMessage]);
        },
        onError: (error: Error) => {
            const errorMessage: Message = {
                id: generateId(),
                role: 'assistant',
                content: `Sorry, I encountered an error: ${error.message}`,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        },
    });

    const sendMessage = useCallback((content: string) => {
        const userMessage: Message = {
            id: generateId(),
            role: 'user',
            content: content.trim(),
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMessage]);
        queryMutation.mutate({ query: content.trim() });
    }, [queryMutation]);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        messages,
        sendMessage,
        clearMessages,
        isLoading: queryMutation.isPending,
        error: queryMutation.error,
    };
}

/**
 * Hook for health check status
 */
export function useHealth() {
    return useQuery({
        queryKey: ['health'],
        queryFn: ragApi.health,
        retry: false,
        refetchInterval: 30000,
    });
}

/**
 * Hook for configuration
 */
export function useConfig() {
    return useQuery({
        queryKey: ['config'],
        queryFn: ragApi.config,
        retry: false,
    });
}

/**
 * Hook for file upload
 */
export function useUpload(onSuccess?: () => void) {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (file: File) => ragApi.upload(file, true),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['health'] });
            onSuccess?.();
        },
    });
}

/**
 * Hook for rebuilding the index
 */
export function useRebuildIndex(onSuccess?: () => void) {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ragApi.rebuildIndex,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['health'] });
            onSuccess?.();
        },
    });
}
