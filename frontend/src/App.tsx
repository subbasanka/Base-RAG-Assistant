import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './contexts/ThemeContext';
import { Sidebar } from './components/Sidebar';
import { Chat } from './components/Chat';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      retry: 1,
    },
  },
});

function App() {
  const handleRebuildComplete = () => {
    // Refetch health status after rebuild
    queryClient.invalidateQueries({ queryKey: ['health'] });
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <div className="flex h-screen">
          <Sidebar onRebuildComplete={handleRebuildComplete} />
          <Chat />
        </div>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
