import os
import json
from typing import Dict, List, Optional
from groq import Groq
import asyncio
import logging
from datetime import datetime
import time
from collections import deque

# Set up minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'tree_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
                    logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s...")
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
    
    def _create_topic_tree_prompt(self, topic: str, depth: int, parent_topic: Optional[str] = None) -> str:
        if depth == 0:
            return f"""Break down the topic '{topic}' into 3-5 main subtopics.
Return ONLY a JSON array of strings, with no additional text or explanation.
Example format: ["Subtopic 1", "Subtopic 2", "Subtopic 3"]"""
        else:
            return f"""Break down the subtopic '{topic}' under '{parent_topic}' into 3-5 key aspects.
Return ONLY a JSON array of strings, with no additional text or explanation.
Example format: ["Aspect 1", "Aspect 2", "Aspect 3"]"""

    async def _get_llm_response(self, prompt: str) -> List[str]:
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait_if_needed()
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-70b-8192",
                        stream=False
                    )
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    # Clean the response to ensure it's valid JSON
                    content = content.strip('`').strip()
                    if content.startswith('json'):
                        content = content[4:].strip()
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract array from text
                    content = content.strip('[]').split(',')
                    return [item.strip().strip('"\'') for item in content]
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"Error occurred. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

    async def _generate_complete_tree(self, topic: str, depth: int = 0, parent_topic: Optional[str] = None) -> Dict:
        if depth >= self.max_depth:
            return {"title": topic, "subtopics": {}}
        
        logger.info(f"Generating subtopics for: {topic}")
        subtopics = await self._get_llm_response(
            self._create_topic_tree_prompt(topic, depth, parent_topic)
        )
        
        subtopic_trees = {}
        for subtopic in subtopics:
            subtopic_trees[subtopic] = await self._generate_complete_tree(subtopic, depth + 1, topic)
        
        return {
            "title": topic,
            "subtopics": subtopic_trees
        }

    async def generate_tree(self, topic: str) -> Dict:
        logger.info(f"Starting tree generation for topic: {topic}")
        tree = await self._generate_complete_tree(topic)
        
        # Save the tree structure
        with open(self.output_file, 'w') as f:
            json.dump(tree, f, indent=2)
        logger.info(f"Tree structure saved to {self.output_file}")
        
        return tree

async def main():
    # Get API key from environment variable
    api_key = "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"
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable")
    
    # Get topic from command line argument or use default
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Data Structures and Algorithms"
    output_file = f"{topic.lower().replace(' ', '_')}_tree.json"
    
    generator = TreeGenerator(api_key, output_file)
    await generator.generate_tree(topic)

if __name__ == "__main__":
    asyncio.run(main()) 