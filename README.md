# LLM Research Assistant

This is a research assistant powered by Groq's LLM that performs deep topic analysis and generates structured research data. It breaks down topics into subtopics and generates comprehensive content for each node.

## Features

- Hierarchical topic breakdown
- Asynchronous processing for faster research
- Concurrent API calls for efficient token usage
- Structured JSON output
- Configurable research depth

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Groq API key:
   - Create a `.env` file in the project root
   - Add your API key: `GROQ_API_KEY=your_api_key_here`

## Usage

1. Run the research assistant:
```bash
python app.py
```

2. The script will:
   - Break down the topic into subtopics
   - Generate detailed content for each subtopic
   - Save the research as a JSON file

## Customization

You can modify the following parameters in the code:
- `max_depth`: Maximum depth of topic breakdown (default: 3)
- `max_workers`: Number of concurrent API calls (default: 5)

## Output Format

The research is saved in a JSON file with the following structure:
```json
{
  "title": "Main Topic",
  "content": "Detailed analysis of the main topic",
  "subtopics": [
    {
      "title": "Subtopic 1",
      "content": "Analysis of subtopic 1",
      "subtopics": [...]
    },
    ...
  ]
}
``` 