import { useState, useRef, useEffect } from 'react';
import { Send, ChevronDown, ChevronUp, Clock, FileText } from 'lucide-react';
import { Button, Textarea, Card, Badge } from './ui';
import { useChat } from '../hooks/useRag';
import type { Message, Source } from '../types/api';

function SourceCard({ source, index }: { source: Source; index: number }) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <Card variant="bordered" className="p-3 text-sm">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center justify-between w-full text-left"
            >
                <div className="flex items-center gap-2">
                    <Badge variant="default">[{index}]</Badge>
                    <span className="font-medium text-gray-700 dark:text-gray-300 truncate max-w-[200px]">
                        {source.source}
                    </span>
                    <span className="text-xs text-gray-500">p.{source.page}</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">{(source.score * 100).toFixed(0)}%</span>
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
            </button>
            {isExpanded && (
                <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-gray-600 dark:text-gray-400 text-xs leading-relaxed">
                        {source.content}
                    </p>
                </div>
            )}
        </Card>
    );
}

function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
            <div className={`max-w-[80%] ${isUser ? 'order-2' : ''}`}>
                <div
                    className={`px-4 py-3 rounded-2xl ${isUser
                            ? 'bg-primary-600 text-white rounded-br-md'
                            : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-md shadow-sm border border-gray-100 dark:border-gray-700'
                        }`}
                >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                </div>

                {/* Sources */}
                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="mt-2 space-y-2">
                        <div className="flex items-center gap-1 text-xs text-gray-500">
                            <FileText className="w-3 h-3" />
                            <span>{message.sources.length} source(s)</span>
                        </div>
                        {message.sources.map((source, idx) => (
                            <SourceCard key={idx} source={source} index={idx + 1} />
                        ))}
                    </div>
                )}

                {/* Metadata */}
                {!isUser && message.latency_ms && (
                    <div className="flex items-center gap-1 mt-1 text-xs text-gray-400">
                        <Clock className="w-3 h-3" />
                        <span>{message.latency_ms.toFixed(0)}ms</span>
                    </div>
                )}
            </div>
        </div>
    );
}

function TypingIndicator() {
    return (
        <div className="flex justify-start animate-fade-in">
            <div className="bg-white dark:bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-md shadow-sm border border-gray-100 dark:border-gray-700">
                <div className="typing-indicator text-gray-400">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    );
}

export function Chat() {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const { messages, sendMessage, isLoading } = useChat();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        sendMessage(input);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
            {/* Header */}
            <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                    💬 Ask Your Documents
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                    Ask questions about your uploaded documents
                </p>
            </header>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 dark:text-gray-400">
                        <div className="text-6xl mb-4">📚</div>
                        <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
                        <p className="text-sm max-w-md">
                            Upload documents using the sidebar, then ask questions about their content.
                        </p>
                    </div>
                ) : (
                    messages.map(message => (
                        <MessageBubble key={message.id} message={message} />
                    ))
                )}
                {isLoading && <TypingIndicator />}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <form onSubmit={handleSubmit} className="flex gap-3">
                    <Textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question about your documents..."
                        rows={1}
                    />
                    <Button type="submit" disabled={!input.trim() || isLoading}>
                        <Send className="w-5 h-5" />
                    </Button>
                </form>
            </div>
        </div>
    );
}
