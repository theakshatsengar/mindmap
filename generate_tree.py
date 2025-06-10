import os
import json
from typing import Dict, List, Optional
from groq import Groq
import asyncio
# import logging
from datetime import datetime
import time
from collections import deque

# Set up minimal logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     handlers=[
#         logging.FileHandler(f'tree_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 25):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = deque(maxlen=requests_per_minute)
        self.lock = asyncio.Lock()
        self.last_request_time = 0
        self.min_request_interval = 2.0
    
    async def wait_if_needed(self):
        async with self.lock:
            current_time = time.time()
            
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            
            if len(self.request_timestamps) >= self.requests_per_minute * 0.8:
                wait_time = 60 - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    # logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
            
            self.request_timestamps.append(current_time)
            self.last_request_time = current_time

class TreeGenerator:
    def __init__(self, api_key: str, output_file: str):
        self.client = Groq(api_key=api_key)
        self.max_depth = 3
        self.rate_limiter = RateLimiter()
        self.output_file = output_file
        self.max_retries = 3
    
    def _create_complete_tree_prompt(self, topic: str) -> str:
        return f"""Generate a detailed mind map for the topic '{topic}' with the following requirements:

1. Create a hierarchical structure with 3-5 levels deep
2. Each node must have:
   - A title (concise and clear)
   - A detailed description (2-4 sentences for non-leaf nodes, 6-8 sentences for leaf nodes)
   - A subtopics object (empty for leaf nodes)

3. Structure Requirements:
   - Root level: One main topic with required number of major subtopics
   - Second level: Each major subtopic should have 1-2 detailed aspects
   - Third level: Only if necessary for very complex topics
   - Leaf nodes should have empty subtopics: {{}}

4. Description Guidelines:
   - Root topic: Overview of the entire subject (2-4 sentences)
   - Major subtopics: Detailed explanation of that aspect (2-4 sentences)
   - Leaf nodes: Comprehensive explanation with specific details (6-8 sentences)

Return the result as a JSON object with this exact structure:
{{
    "title": "Main Topic",
    "description": "2-4 sentence overview of the main topic",
    "subtopics": {{
        "Major Subtopic 1": {{
            "title": "Major Subtopic 1",
            "description": "2-4 sentence detailed explanation of this subtopic",
            "subtopics": {{
                "Detailed Aspect 1": {{
                    "title": "Detailed Aspect 1",
                    "description": "6-8 sentence comprehensive explanation with specific details",
                    "subtopics": {{}}
                }}
            }}
        }},
        "Major Subtopic 2": {{
            "title": "Major Subtopic 2",
            "description": "2-4 sentence detailed explanation of this subtopic",
            "subtopics": {{
                "Detailed Aspect 1": {{
                    "title": "Detailed Aspect 1",
                    "description": "6-8 sentence comprehensive explanation with specific details",
                    "subtopics": {{}}
                }},
                "Detailed Aspect 2": {{
                    "title": "Detailed Aspect 2",
                    "description": "6-8 sentence comprehensive explanation with specific details",
                    "subtopics": {{}}
                }}
            }}
        }}
    }}
}}

IMPORTANT:
1. Return ONLY the JSON object, with no additional text or formatting
2. The response must start with {{ and end with }}
3. Maintain consistent indentation
4. Ensure all descriptions are properly detailed and informative
5. Keep the structure clean and well-organized"""

    async def _get_llm_response(self, prompt: str) -> Dict:
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait_if_needed()
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-70b-8192",
                        temperature=0.7,
                        max_tokens=8192
                    )
                )
                
                content = response.choices[0].message.content.strip()
                print("\nRaw API Response:", content)  # Log the raw response
                
                if not content:
                    raise ValueError("Empty response received from API")
                
                try:
                    # Clean the response to ensure it's valid JSON
                    content = content.strip('`').strip()
                    if content.startswith('json'):
                        content = content[4:].strip()
                    
                    # Try to find JSON content if it's wrapped in other text
                    if '{' in content and '}' in content:
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        content = content[start:end]
                    
                    print("\nCleaned JSON content:", content)  # Log the cleaned content
                    
                    parsed_json = json.loads(content)
                    
                    # Validate the JSON structure
                    if not isinstance(parsed_json, dict):
                        raise ValueError("Response is not a JSON object")
                    if 'title' not in parsed_json:
                        raise ValueError("Response missing 'title' field")
                    if 'subtopics' not in parsed_json:
                        raise ValueError("Response missing 'subtopics' field")
                    
                    return parsed_json
                    
                except json.JSONDecodeError as e:
                    print(f"\nJSON parsing error: {str(e)}")
                    print(f"Failed to parse content: {content}")
                    if attempt < self.max_retries - 1:
                        continue
                    raise ValueError(f"Invalid JSON response: {str(e)}")
                
            except Exception as e:
                print(f"\nError in API call: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    print(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise ValueError(f"Failed to generate mind map: {str(e)}")

    async def generate_tree(self, topic: str) -> Dict:
        # logger.info(f"Starting tree generation for topic: {topic}")
        
        # Generate the complete tree in a single API call
        tree = await self._get_llm_response(self._create_complete_tree_prompt(topic))
        
        # Save the tree structure
        with open(self.output_file, 'w') as f:
            json.dump(tree, f, indent=2)
        # logger.info(f"Tree structure saved to {self.output_file}")
        
        return tree

async def main():
    # Get API key from environment variable
    api_key = "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"
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable")
    
    # Get topic from command line argument or use default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Data Structures and Algorithms"
    output_file = "tree.json"
    
    generator = TreeGenerator(api_key, output_file)
    await generator.generate_tree(topic)

if __name__ == "__main__":
    asyncio.run(main()) 