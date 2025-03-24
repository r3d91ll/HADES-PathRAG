"""
LLM-powered code analysis module for HADES-PathRAG.

Uses Qwen2.5 Coder or other code-specialized LLMs to perform deep semantic analysis
of code elements, extract relationships, and generate meaningful descriptions.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Analysis prompt templates
CODE_ANALYSIS_PROMPT = """
You are a code analysis expert. Analyze the following code and extract its structure.
Focus on identifying key components and their relationships:

1. Module imports and their sources (standard lib, third-party, local)
2. Classes with their methods and attributes
3. Functions with their parameters, return types, and docstrings
4. Complex relationships like inheritance, composition, function calls
5. Any public API endpoints or entry points

Code to analyze:
```python
{code}
```

Output your analysis as a JSON object with the following structure:
{
    "imports": [
        {"module": "module_name", "aliases": ["alias"], "source": "std_lib|third_party|local", "used_by": ["element_names"]}
    ],
    "classes": [
        {
            "name": "ClassName",
            "inherits_from": ["ParentClass"],
            "attributes": [{"name": "attr_name", "type": "type_if_known"}],
            "methods": [
                {
                    "name": "method_name", 
                    "params": [{"name": "param_name", "type": "type_if_known"}],
                    "return_type": "return_type_if_known",
                    "description": "Short functional description",
                    "calls": ["other_function_names"],
                    "access": "public|protected|private"
                }
            ],
            "description": "What this class represents and does"
        }
    ],
    "functions": [
        {
            "name": "function_name",
            "params": [{"name": "param_name", "type": "type_if_known"}],
            "return_type": "return_type_if_known",
            "description": "Short functional description",
            "calls": ["other_function_names"],
            "complexity": "low|medium|high"
        }
    ],
    "entry_points": ["possible_entry_point_functions"],
    "global_vars": [{"name": "var_name", "type": "type_if_known", "purpose": "brief description"}]
}

BE SURE TO OUTPUT VALID JSON that can be parsed with json.loads()
"""

RELATIONSHIP_EXTRACTION_PROMPT = """
You are analyzing the relationships between code elements in a Python project.
Given the extracted information from two different files, identify all relationships between them:

File 1: {file1_path}
```
{file1_info}
```

File 2: {file2_path}
```
{file2_info}
```

Identify all explicit and implicit relationships between these files, including:
1. Import relationships
2. Inheritance relationships
3. Function/method calls
4. Composition/aggregation relationships
5. Extension of functionality

Output JSON with these relationships:
{
    "relationships": [
        {
            "type": "import|inheritance|call|composition|extension",
            "from_file": "file_path",
            "to_file": "file_path",
            "from_element": "element_name",
            "to_element": "element_name",
            "description": "Brief description of the relationship",
            "confidence": 0.0-1.0
        }
    ]
}

BE SURE TO OUTPUT VALID JSON that can be parsed with json.loads()
"""

DOCUMENTATION_GENERATION_PROMPT = """
Generate comprehensive documentation for the following code element.
Focus on explaining:
1. Purpose and functionality
2. Parameters and return values
3. Usage examples
4. Internal workings where relevant
5. Relationships with other components

Code element:
```python
{code}
```

Output JSON with documentation:
{
    "name": "element_name",
    "type": "module|class|function|method",
    "summary": "One-line summary",
    "description": "Detailed multi-paragraph description",
    "params": [{"name": "param_name", "type": "type", "description": "what this parameter does"}],
    "returns": {"type": "return_type", "description": "what is returned"},
    "exceptions": [{"type": "exception_type", "description": "when this is raised"}],
    "examples": ["code example showing how to use this element"],
    "related_elements": ["names of related code elements"]
}

BE SURE TO OUTPUT VALID JSON that can be parsed with json.loads()
"""

class LLMCodeAnalyzer:
    """LLM-powered code analyzer using Ollama models."""
    
    def __init__(self, model_name: str = "qwen2.5-128k"):
        """Initialize the analyzer with a specified code-specialized LLM.
        
        Args:
            model_name: Name of the Ollama model to use for code analysis
        """
        self.model_name = model_name
        logger.info(f"Initializing LLM Code Analyzer with model: {model_name}")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def _query_model(self, prompt: str, temperature: float = 0.1) -> str:
        """Query the LLM with a prompt and return the response.
        
        Args:
            prompt: The full prompt to send to the model
            temperature: Temperature setting for generation (lower = more deterministic)
            
        Returns:
            The model's response as a string
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 4096  # Allow longer outputs for complex analyses
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            raise
    
    def analyze_code_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Perform deep semantic analysis of a code file.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Dictionary containing structured analysis of the code
        """
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # Skip empty files or files that are too large
            if not code_content.strip() or len(code_content) > 100000:
                logger.warning(f"Skipping file {file_path}: {'Empty' if not code_content.strip() else 'Too large'}")
                return None
                
            # Create the analysis prompt
            prompt = CODE_ANALYSIS_PROMPT.format(code=code_content)
            
            # Query the model
            response = self._query_model(prompt)
            
            # Extract JSON from response
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
                
                # Clean up any markdown or extra text
                json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
                
                # Parse the JSON response
                analysis = json.loads(json_str)
                
                # Add metadata
                analysis["file_path"] = str(file_path)
                return analysis
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON from model response for {file_path}: {je}")
                logger.debug(f"Response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def extract_relationships(self, file1_info: Dict[str, Any], file2_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships between two analyzed files.
        
        Args:
            file1_info: Analysis output for first file
            file2_info: Analysis output for second file
            
        Returns:
            List of relationship dictionaries
        """
        try:
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                file1_path=file1_info.get("file_path", "Unknown"),
                file1_info=json.dumps(file1_info, indent=2),
                file2_path=file2_info.get("file_path", "Unknown"),
                file2_info=json.dumps(file2_info, indent=2)
            )
            
            response = self._query_model(prompt)
            
            # Extract JSON from response
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
                
                # Clean up any markdown or extra text
                json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
                
                # Parse the JSON response
                relationship_data = json.loads(json_str)
                return relationship_data.get("relationships", [])
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from relationship extraction")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def generate_documentation(self, code_element: str, element_type: str) -> Optional[Dict[str, Any]]:
        """Generate rich documentation for a code element.
        
        Args:
            code_element: The full code of the element (function, class, etc.)
            element_type: Type of the element ("function", "class", etc.)
            
        Returns:
            Dictionary containing generated documentation
        """
        try:
            prompt = DOCUMENTATION_GENERATION_PROMPT.format(code=code_element)
            
            response = self._query_model(prompt, temperature=0.2)
            
            # Extract JSON from response
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
                
                # Clean up any markdown or extra text
                json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
                
                # Parse the JSON response
                doc_data = json.loads(json_str)
                return doc_data
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from documentation generation")
                return None
                
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return None
