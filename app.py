from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import os
import json
from urllib.parse import parse_qs, urlparse
import asyncio
from generate_tree import TreeGenerator

class MindMapHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve index.html as the default page
        if self.path == '/':
            self.path = '/index.html'
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            topic = data.get('topic')

            if not topic:
                self.send_error(400, "Topic is required")
                return

            try:
                # Generate the tree with consistent filename
                output_file = "tree.json"
                generator = TreeGenerator("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", output_file)
                asyncio.run(generator.generate_tree(topic))

                # Save the topic for reference
                with open('current_topic.txt', 'w') as f:
                    f.write(topic)

                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'redirect': f'/mindmap.html?topic={topic}'
                }).encode())
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MindMapHandler)
    print(f"Server running at http://localhost:{port}")
    webbrowser.open(f'http://localhost:{port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 