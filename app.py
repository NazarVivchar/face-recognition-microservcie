import socketserver
from src.api.HTTPRequestHandler import HTTPRequestHandler

PORT = 8082

with socketserver.TCPServer(("", PORT), HTTPRequestHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
