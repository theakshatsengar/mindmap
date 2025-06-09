from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import os

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Server running at http://localhost:{port}")
    webbrowser.open(f'http://localhost:{port}/mindmap.html')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 