import { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ragApi } from '../api/client';
import type { Message, Source } from '../types/api';

// Generate unique IDs
const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

/**
 * Hook for managing chat messages with streaming responses
 */
export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [streamingContent, setStreamingContent] = useState('');
    const [error, setError] = useState<Error | null>(null);

    const sendMessage = useCallback(async (content: string) => {
        const userMessage: Message = {
            id: generateId(),
            role: 'user',
            content: content.trim(),
            timestamp: new Date(),
        };

        const assistantMessageId = generateId();

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);
        setStreamingContent('');
        setError(null);

        try {
            let fullContent = '';
            let sources: Source[] = [];
            let latency_ms: number | undefined;

            for await (const chunk of ragApi.queryStream({ query: content.trim() })) {
                if (chunk.error) {
                    throw new Error(chunk.error);
                }

                if (chunk.chunk) {
                    fullContent += chunk.chunk;
                    setStreamingContent(fullContent);
                }

                if (chunk.done && chunk.sources) {
                    sources = chunk.sources;
                    latency_ms = chunk.latency_ms;
                }
            }

            const assistantMessage: Message = {
                id: assistantMessageId,
                role: 'assistant',
                content: fullContent,
                sources,
                latency_ms,
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (e) {
            const errorMessage: Message = {
                id: assistantMessageId,
                role: 'assistant',
                content: `Sorry, I encountered an error: ${e instanceof Error ? e.message : 'Unknown error'}`,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
            setError(e instanceof Error ? e : new Error('Unknown error'));
        } finally {
            setIsLoading(false);
            setStreamingContent('');
        }
    }, []);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        messages,
        sendMessage,
        clearMessages,
        isLoading,
        streamingContent,
        error,
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
