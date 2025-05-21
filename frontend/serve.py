import http.server
import socketserver
import os
import sys

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[SERVER] {format % args}")

def serve():
    try:
        handler = CORSHTTPRequestHandler
        httpd = socketserver.TCPServer(("", PORT), handler)

        print(f"\n[SERVER] Serving at http://localhost:{PORT}")
        print("[SERVER] Press Ctrl+C to stop the server\n")
        
        # Open the browser automatically (optional)
        # import webbrowser
        # webbrowser.open(f'http://localhost:{PORT}')
        
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[SERVER] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    serve() 