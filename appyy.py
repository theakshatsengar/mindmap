import os
import json
from typing import Dict, List, Optional
from groq import Groq
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import time
from collections import deque
import random

# Set up minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'research_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self):
        self.total_topics = 0
        self.completed_topics = 0
        self.current_path = []
    
    def update_progress(self, topic: str):
        self.current_path.append(topic)
        path_str = " > ".join(self.current_path)
        logger.info(f"Researching: {path_str}")
    
    def complete_topic(self):
        self.completed_topics += 1
        if self.current_path:
            self.current_path.pop()
        logger.info(f"Progress: {self.completed_topics}/{self.total_topics} topics completed")

class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps = deque(maxlen=requests_per_minute)
        self.token_timestamps = deque(maxlen=tokens_per_minute)
        self.lock = asyncio.Lock()
        self.last_request_time = 0
        self.min_request_interval = 2.0
    
    async def wait_if_needed(self, estimated_tokens: int):
        async with self.lock:
            current_time = time.time()
            
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                await asyncio.sleep(wait_time)
                current_time = time.time()
            
            while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            while self.token_timestamps and current_time - self.token_timestamps[0] > 60:
                self.token_timestamps.popleft()
            
            if len(self.request_timestamps) >= self.requests_per_minute * 0.8:
                wait_time = 60 - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
            
            if len(self.token_timestamps) >= self.tokens_per_minute * 0.8:
                wait_time = 60 - (current_time - self.token_timestamps[0])
                if wait_time > 0:
                    logger.info(f"Token limit approaching. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
            
            self.request_timestamps.append(current_time)
            self.token_timestamps.append(current_time)
            self.last_request_time = current_time

@dataclass
class ResearchNode:
    title: str
    content: str
    subtopics: List['ResearchNode']
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content,
            "subtopics": [subtopic.to_dict() for subtopic in self.subtopics]
        }

class ResearchAssistant:
    def __init__(self, api_key: str, output_file: str):
        self.client = Groq(api_key=api_key)
        self.max_depth = 3
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.rate_limiter = RateLimiter(requests_per_minute=25, tokens_per_minute=5000)
        self.progress = ProgressTracker()
        self.max_retries = 3
        self.output_file = output_file
        self.tree_file = output_file.replace('.json', '_tree.json')
        self.research_data = {}
        self.save_lock = asyncio.Lock()
    
    def _create_topic_tree_prompt(self, topic: str, depth: int, parent_topic: Optional[str] = None) -> str:
        if depth == 0:
            return f"""Break down the topic '{topic}' into 3-5 main subtopics.
Return ONLY a JSON array of strings, with no additional text or explanation.
Example format: ["Subtopic 1", "Subtopic 2", "Subtopic 3"]"""
        else:
            return f"""Break down the subtopic '{topic}' under '{parent_topic}' into 3-5 key aspects.
Return ONLY a JSON array of strings, with no additional text or explanation.
Example format: ["Aspect 1", "Aspect 2", "Aspect 3"]"""

    def _create_content_prompt(self, topic: str, depth: int, parent_topic: Optional[str] = None) -> str:
        return f"""Provide a comprehensive analysis of the topic '{topic}'.
Focus on key concepts, important details, and relevant information.
Return ONLY the analysis text, with no additional formatting or explanation."""

    async def _get_llm_response(self, prompt: str, is_json: bool = False) -> str:
        estimated_tokens = len(prompt) // 3
        
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait_if_needed(estimated_tokens)
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-70b-8192",
                        stream=False
                    )
                )
                
                content = response.choices[0].message.content.strip()
                
                if is_json:
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
                return content
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Error occurred. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

    async def _generate_complete_tree(self, topic: str, depth: int = 0, parent_topic: Optional[str] = None) -> Dict:
        if depth >= self.max_depth:
            return {"title": topic, "subtopics": {}}
        
        subtopics = await self._get_llm_response(
            self._create_topic_tree_prompt(topic, depth, parent_topic),
            is_json=True
        )
        
        subtopic_trees = {}
        for subtopic in subtopics:
            subtopic_trees[subtopic] = await self._generate_complete_tree(subtopic, depth + 1, topic)
        
        return {
            "title": topic,
            "subtopics": subtopic_trees
        }

    async def _save_tree_structure(self, tree: Dict):
        with open(self.tree_file, 'w') as f:
            json.dump(tree, f, indent=2)
        logger.info(f"Complete topic tree saved to {self.tree_file}")

    async def _save_progress(self, node: ResearchNode, path: List[str]):
        async with self.save_lock:
            current = self.research_data
            for part in path[:-1]:
                if part not in current:
                    current[part] = {"title": part, "content": "", "subtopics": {}}
                current = current[part]["subtopics"]
            
            current[path[-1]] = {
                "title": node.title,
                "content": node.content,
                "subtopics": {st.title: st.to_dict() for st in node.subtopics}
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(self.research_data, f, indent=2)
            logger.info(f"Progress saved to {self.output_file}")

    async def _save_topic_tree(self, topic: str, subtopics: List[str], path: List[str]):
        async with self.save_lock:
            current = self.research_data
            for part in path[:-1]:
                if part not in current:
                    current[part] = {"title": part, "content": "", "subtopics": {}}
                current = current[part]["subtopics"]
            
            # Create placeholder structure for the topic
            current[path[-1]] = {
                "title": topic,
                "content": "",
                "subtopics": {st: {"title": st, "content": "", "subtopics": {}} for st in subtopics}
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(self.research_data, f, indent=2)
            logger.info(f"Topic tree saved to {self.output_file}")

    async def _research_topic(self, topic: str, depth: int = 0, parent_topic: Optional[str] = None, path: List[str] = None) -> ResearchNode:
        if path is None:
            path = [topic]
        
        self.progress.update_progress(topic)
        
        if depth >= self.max_depth:
            content = await self._get_llm_response(self._create_content_prompt(topic, depth, parent_topic))
            node = ResearchNode(title=topic, content=content, subtopics=[])
            await self._save_progress(node, path)
            self.progress.complete_topic()
            return node
        
        # First generate the topic tree
        subtopics = await self._get_llm_response(
            self._create_topic_tree_prompt(topic, depth, parent_topic),
            is_json=True
        )
        self.progress.total_topics += len(subtopics)
        
        # Save the topic tree structure immediately
        await self._save_topic_tree(topic, subtopics, path)
        
        # Then get content for current topic
        content = await self._get_llm_response(self._create_content_prompt(topic, depth, parent_topic))
        
        # Recursively research subtopics
        subtopic_nodes = []
        for subtopic in subtopics:
            subtopic_path = path + [subtopic]
            node = await self._research_topic(subtopic, depth + 1, topic, subtopic_path)
            subtopic_nodes.append(node)
        
        node = ResearchNode(title=topic, content=content, subtopics=subtopic_nodes)
        await self._save_progress(node, path)
        self.progress.complete_topic()
        return node

    async def research(self, topic: str) -> Dict:
        logger.info(f"Starting research on: {topic}")
        start_time = datetime.now()
        
        try:
            # First generate and save the complete tree structure
            logger.info("Generating complete topic tree...")
            tree = await self._generate_complete_tree(topic)
            await self._save_tree_structure(tree)
            logger.info("Topic tree generation completed")
            
            # Now start the research process
            root_node = await self._research_topic(topic)
            result = root_node.to_dict()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Research completed in {duration:.1f}s")
            
            return result
        except Exception as e:
            logger.info(f"Research failed: {str(e)}")
            raise

async def main():
    api_key = "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"
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    topic = "data structure and algorithm for frontend dev role in startups"
    output_file = f"research_{topic.lower().replace(' ', '_')}.json"
    
    assistant = ResearchAssistant(api_key, output_file)
    result = await assistant.research(topic)
    
    logger.info(f"Final results saved to: {output_file}")
    logger.info(f"Topic tree saved to: {assistant.tree_file}")

if __name__ == "__main__":
    asyncio.run(main())
