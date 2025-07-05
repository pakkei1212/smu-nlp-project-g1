# src/image_results_manager.py
"""
Simple results manager for image RAG queries
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ImageResultsManager:
    """
    Simple manager for saving and loading image RAG results
    """
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the image results manager
        
        Args:
            image_results_dir: Directory to store results (defaults to 'output/rag_results')
        """
        self.results = {}
        if results_dir is None:
            # Use same directory as your existing text results
            self.results_dir = Path("output/rag_results")
        else:
            self.results_dir = Path(results_dir)

        # Ensure the directory exists    
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    
    def add_results(self, image_result_dict: Dict[str, Any]) -> None:
        """
        Add results to the manager
        
        Args:
            image_result_dict: Dictionary of image query results
        """
        self.results.update(image_result_dict)
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save all results to a JSON file
        
        Args:
            filename: Optional filename (defaults to 'image_rag_results_YYYYMMDD.json')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"image_rag_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        return filepath
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load results from a JSON file
        
        Args:
            filename: Name of file to load
            
        Returns:
            Loaded results dictionary
        """
        filepath = self.results_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
        
        self.results.update(loaded_results)
        return loaded_results
    
    def clear_results(self) -> None:
        """
        Clear all results from memory
        """
        self.results.clear()
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a simple summary of current results
        
        Returns:
            Summary dictionary
        """
        summary = {
            "total_queries": len(self.results),
            "query_types": {},
            "sources": set()
        }
        
        for query_id, result in self.results.items():
            # Determine type
            if 'error' in result:
                query_type = "error"
            elif 'results' in result:
                query_type = "search"
                # Collect sources from search results
                for res in result.get('results', []):
                    if res.get('source_file'):
                        summary["sources"].add(res['source_file'])
            elif 'answer' in result:
                query_type = "direct_query"
                # Collect source from direct query
                image_source = result.get('image_source', {})
                if image_source.get('file'):
                    summary["sources"].add(image_source['file'])
            else:
                query_type = "unknown"
            
            summary["query_types"][query_type] = summary["query_types"].get(query_type, 0) + 1
        
        # Convert set to list for JSON serialization
        summary["sources"] = list(summary["sources"])
        
        return summary
    
    def list_saved_files(self) -> list:
        """
        List all saved result files in the results directory
        
        Returns:
            List of filenames
        """
        return [f.name for f in self.results_dir.glob("*.json")]