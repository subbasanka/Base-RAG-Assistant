import { Sun, Moon, Settings, RefreshCw, FileText, Upload } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { FileUpload } from './FileUpload';
import { Button, Badge } from './ui';
import { useState } from 'react';
import { useHealth, useConfig, useRebuildIndex } from '../hooks/useRag';

interface SidebarProps {
    onRebuildComplete?: () => void;
}

export function Sidebar({ onRebuildComplete }: SidebarProps) {
    const { theme, toggleTheme } = useTheme();
    const [activeTab, setActiveTab] = useState<'chat' | 'upload'>('chat');

    const { data: health, isSuccess: healthSuccess } = useHealth();
    const { data: config } = useConfig();
    const rebuildMutation = useRebuildIndex(onRebuildComplete);

    return (
        <aside className="w-72 h-screen bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                    <h1 className="text-xl font-bold text-primary-600 dark:text-primary-400 flex items-center gap-2">
                        📚 RAG Assistant
                    </h1>
                    <Button variant="ghost" size="sm" onClick={toggleTheme}>
                        {theme === 'dark' ? (
                            <Sun className="w-5 h-5 text-yellow-500" />
                        ) : (
                            <Moon className="w-5 h-5 text-gray-600" />
                        )}
                    </Button>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200 dark:border-gray-700">
                <button
                    onClick={() => setActiveTab('chat')}
                    className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'chat'
                            ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-600 dark:border-primary-400'
                            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                        }`}
                >
                    <Settings className="w-4 h-4 inline-block mr-1" />
                    Config
                </button>
                <button
                    onClick={() => setActiveTab('upload')}
                    className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'upload'
                            ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-600 dark:border-primary-400'
                            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                        }`}
                >
                    <Upload className="w-4 h-4 inline-block mr-1" />
                    Upload
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
                {activeTab === 'upload' ? (
                    <FileUpload onUploadComplete={onRebuildComplete} />
                ) : (
                    <div className="space-y-4">
                        {/* Status */}
                        <div className="space-y-2">
                            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Status</h3>
                            <div className="flex items-center gap-2">
                                <Badge variant={healthSuccess ? 'success' : 'error'}>
                                    {healthSuccess ? 'Connected' : 'Disconnected'}
                                </Badge>
                            </div>
                            {health?.index_loaded !== undefined && (
                                <div className="flex items-center gap-2">
                                    <FileText className="w-4 h-4 text-gray-500" />
                                    <span className="text-sm text-gray-600 dark:text-gray-400">
                                        Index: {health.index_loaded ? 'Loaded' : 'Not loaded'}
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Config Info */}
                        {config && (
                            <div className="space-y-2">
                                <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Configuration</h3>
                                <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                                    <p><span className="font-medium">Model:</span> {config.llm.model_name}</p>
                                    <p><span className="font-medium">Embeddings:</span> {config.embeddings.provider}</p>
                                    <p><span className="font-medium">Top-K:</span> {config.retrieval.top_k}</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Footer Actions */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                <Button
                    variant="secondary"
                    onClick={() => rebuildMutation.mutate()}
                    isLoading={rebuildMutation.isPending}
                    className="w-full"
                >
                    <RefreshCw className={`w-4 h-4 mr-2 ${rebuildMutation.isPending ? 'animate-spin' : ''}`} />
                    Rebuild Index
                </Button>
            </div>
        </aside>
    );
}
