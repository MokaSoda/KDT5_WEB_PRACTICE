import http.server as server


PORT = 8000

Handler = server.CGIHTTPRequestHandler

httpd = server.HTTPServer(("", PORT), Handler)
print("serving at port", PORT)
httpd.serve_forever()

