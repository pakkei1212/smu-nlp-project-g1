# src/rag_results_manager.py
"""
Module for managing and displaying RAG query results.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from IPython.display import display, Markdown, HTML
import json
import datetime
import os
from pathlib import Path

class RAGResultsManager:
    """
    Manager for RAG query results, providing storage and display functionality
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the RAG results manager
        
        Args:
            results_dir: Directory to store results (defaults to 'output/rag_results')
        """
        self.results = {}
        self.results_dir = results_dir or Path('output/rag_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: Dict[str, Any], result_id: Optional[str] = None) -> str:
        """
        Add a result to the manager
        
        Args:
            result: RAG query result
            result_id: Optional ID for the result (defaults to timestamp)
            
        Returns:
            Result ID
        """
        # Generate ID if not provided
        if result_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            query_words = result.get('query', '')[:20].replace(' ', '_')
            result_id = f"{query_words}_{timestamp}"
        
        # Add timestamp if not present
        if 'timestamp' not in result:
            result['timestamp'] = datetime.datetime.now().isoformat()
        
        # Store result
        self.results[result_id] = result
        return result_id
    
    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a result by ID
        
        Args:
            result_id: Result ID
            
        Returns:
            Result dictionary or None if not found
        """
        return self.results.get(result_id)
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save all results to a JSON file
        
        Args:
            filename: Optional filename (defaults to 'rag_results_YYYYMMDD.json')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"rag_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_results(self, filepath: Path) -> Dict[str, Any]:
        """
        Load results from a JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded results
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
        
        self.results.update(loaded_results)
        return self.results
    
    def display_result(self, result: Dict[str, Any], max_context_length: int = 200) -> None:
        """
        Display a single RAG result in a readable format
        
        Args:
            result: RAG result dictionary
            max_context_length: Maximum length of context preview
        """
        if not result:
            display(Markdown("### ❌ No result to display"))
            return
        
        query = result.get('query', 'Unknown query')
        answer = result.get('answer', 'No answer generated')
        contexts = result.get('contexts', [])
        
        # Format contexts
        context_md = ""
        for i, ctx in enumerate(contexts):
            text = ctx.get('text', '')
            preview = text[:max_context_length] + "..." if len(text) > max_context_length else text
            metadata = ctx.get('metadata', {})
            relevance = ctx.get('relevance', 0.0)
            
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'Unknown')
            content_type = metadata.get('content_type', 'Unknown')
            
            context_md += f"#### Context {i+1}\n"
            context_md += f"- **Source**: {source}, **Page**: {page}, **Type**: {content_type}\n"
            context_md += f"- **Relevance**: {relevance:.4f}\n"
            context_md += f"- **Preview**: {preview}\n\n"
        
        # Create markdown
        markdown_content = f"""
## Query Result

### Query
{query}

### Answer
{answer}

### Context Sources
{context_md}
"""
        display(Markdown(markdown_content))
    
    def display_all_results(self, max_context_length: int = 100) -> None:
        """
        Display all results
        
        Args:
            max_context_length: Maximum length of context preview
        """
        for result_id, result in self.results.items():
            display(Markdown(f"## Result: {result_id}"))
            self.display_result(result, max_context_length)
            display(Markdown("---"))
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all results
        
        Returns:
            DataFrame with summary
        """
        summary_data = []
        
        for result_id, result in self.results.items():
            query = result.get('query', 'Unknown')
            answer = result.get('answer', 'No answer')
            
            # Get timestamp
            timestamp = result.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = timestamp
            else:
                formatted_time = str(timestamp)
            
            # Count contexts
            num_contexts = len(result.get('contexts', []))
            
            # Calculate average relevance
            relevances = [ctx.get('relevance', 0.0) for ctx in result.get('contexts', [])]
            avg_relevance = sum(relevances) / len(relevances) if relevances else 0.0
            
            # Preview answer
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            
            summary_data.append({
                'ID': result_id,
                'Query': query,
                'Answer Preview': answer_preview,
                'Contexts': num_contexts,
                'Avg. Relevance': avg_relevance,
                'Timestamp': formatted_time
            })
        
        return pd.DataFrame(summary_data)
    
    def display_summary_table(self) -> None:
        """
        Display a summary table of all results
        """
        df = self.create_summary_table()
        
        if len(df) == 0:
            display(Markdown("### No results to display"))
            return
        
        # Style the DataFrame
        styled_df = df.style.set_properties(**{
            'white-space': 'pre-wrap',
            'max-width': '200px',
            'overflow': 'hidden',
            'text-overflow': 'ellipsis'
        })
        
        # Display as HTML
        display(HTML(styled_df.to_html()))
    
    def display_markdown_table(self) -> None:
        """
        Display a summary table in Markdown format
        """
        df = self.create_summary_table()
        
        if len(df) == 0:
            display(Markdown("### No results to display"))
            return
        
        # Convert to markdown
        markdown_table = df.to_markdown(index=False)
        display(Markdown(markdown_table))