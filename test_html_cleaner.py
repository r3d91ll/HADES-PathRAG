"""
Temporary script to test the HTML cleaner functionality.
"""
from src.docproc.adapters.docling_adapter import DoclingAdapter
from pathlib import Path

def main():
    adapter = DoclingAdapter()
    result = adapter.process(Path('/home/todd/ML-Lab/Olympus/HADES-PathRAG/data/langchain_docling.html'))
    
    # Print stats
    print('Original content length:', len(result['raw_content']), 'bytes')
    print('Cleaned content length:', len(result['content']), 'bytes')
    print('Content reduction:', round((1 - len(result['content'])/len(result['raw_content'])) * 100, 2), '%')
    
    # Print metadata
    print('\nMetadata extracted:')
    for key, value in result['metadata'].items():
        print(f'- {key}: {type(value).__name__}')
    
    # Print sample of cleaned content
    print('\nSample of cleaned content (first 300 chars):')
    print(result['content'][:300])

if __name__ == "__main__":
    main()
