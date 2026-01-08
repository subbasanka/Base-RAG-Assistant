import { useState, useCallback } from 'react';
import { Upload, X, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { Button, Card } from './ui';
import { useUpload } from '../hooks/useRag';

interface FileUploadProps {
    onUploadComplete?: () => void;
}

export function FileUpload({ onUploadComplete }: FileUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const uploadMutation = useUpload(() => {
        setSelectedFile(null);
        onUploadComplete?.();
    });

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && isValidFile(file)) {
            setSelectedFile(file);
        }
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && isValidFile(file)) {
            setSelectedFile(file);
        }
    };

    const isValidFile = (file: File): boolean => {
        const validTypes = ['.pdf', '.txt', '.md'];
        return validTypes.some(ext => file.name.toLowerCase().endsWith(ext));
    };

    const handleUpload = () => {
        if (selectedFile) {
            uploadMutation.mutate(selectedFile);
        }
    };

    const formatFileSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    return (
        <div className="space-y-3">
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`
          relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200
          ${isDragging
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-primary-400'
                    }
        `}
            >
                <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Drop files here or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-1">PDF, TXT, MD</p>
            </div>

            {selectedFile && (
                <Card className="animate-fade-in">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <FileText className="w-5 h-5 text-primary-500" />
                            <div>
                                <p className="text-sm font-medium truncate max-w-[150px]">
                                    {selectedFile.name}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {formatFileSize(selectedFile.size)}
                                </p>
                            </div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedFile(null)}>
                            <X className="w-4 h-4" />
                        </Button>
                    </div>

                    <Button
                        onClick={handleUpload}
                        isLoading={uploadMutation.isPending}
                        className="w-full mt-3"
                    >
                        Upload & Index
                    </Button>
                </Card>
            )}

            {uploadMutation.isSuccess && (
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400 text-sm animate-fade-in">
                    <CheckCircle className="w-4 h-4" />
                    <span>Uploaded successfully!</span>
                </div>
            )}

            {uploadMutation.isError && (
                <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-sm animate-fade-in">
                    <AlertCircle className="w-4 h-4" />
                    <span>Upload failed. Try again.</span>
                </div>
            )}
        </div>
    );
}
