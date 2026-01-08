import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, InputHTMLAttributes, TextareaHTMLAttributes, HTMLAttributes } from 'react';
import { cn } from '../../utils/cn';

// Button Component
interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', isLoading, disabled, children, ...props }, ref) => {
        const baseStyles = 'inline-flex items-center justify-center font-medium rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';

        const variants = {
            primary: 'bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500',
            secondary: 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-900 dark:text-white focus:ring-gray-500',
            ghost: 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 focus:ring-gray-500',
        };

        const sizes = {
            sm: 'px-3 py-1.5 text-sm',
            md: 'px-4 py-2 text-sm',
            lg: 'px-6 py-3 text-base',
        };

        return (
            <button
                ref={ref}
                className={cn(baseStyles, variants[variant], sizes[size], className)}
                disabled={disabled || isLoading}
                {...props}
            >
                {isLoading && (
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                )}
                {children}
            </button>
        );
    }
);
Button.displayName = 'Button';

// Input Component
interface InputProps extends InputHTMLAttributes<HTMLInputElement> { }

export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ className, ...props }, ref) => {
        return (
            <input
                ref={ref}
                className={cn(
                    'w-full px-4 py-3 bg-gray-100 dark:bg-gray-700 border-0 rounded-xl',
                    'focus:outline-none focus:ring-2 focus:ring-primary-500',
                    'text-gray-900 dark:text-white placeholder-gray-500',
                    className
                )}
                {...props}
            />
        );
    }
);
Input.displayName = 'Input';

// Textarea Component
interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> { }

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ className, ...props }, ref) => {
        return (
            <textarea
                ref={ref}
                className={cn(
                    'w-full px-4 py-3 bg-gray-100 dark:bg-gray-700 border-0 rounded-xl resize-none',
                    'focus:outline-none focus:ring-2 focus:ring-primary-500',
                    'text-gray-900 dark:text-white placeholder-gray-500',
                    className
                )}
                {...props}
            />
        );
    }
);
Textarea.displayName = 'Textarea';

// Card Component
interface CardProps extends HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'bordered';
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, variant = 'default', ...props }, ref) => {
        const variants = {
            default: 'bg-white dark:bg-gray-800 shadow-sm',
            bordered: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
        };

        return (
            <div
                ref={ref}
                className={cn('rounded-xl p-4', variants[variant], className)}
                {...props}
            />
        );
    }
);
Card.displayName = 'Card';

// Badge Component
interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
    variant?: 'default' | 'success' | 'warning' | 'error';
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
    ({ className, variant = 'default', ...props }, ref) => {
        const variants = {
            default: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
            success: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400',
            warning: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400',
            error: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400',
        };

        return (
            <span
                ref={ref}
                className={cn('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium', variants[variant], className)}
                {...props}
            />
        );
    }
);
Badge.displayName = 'Badge';
