import json
from http.server import BaseHTTPRequestHandler
from src.services.FaceVectorService import calc_embs


def converter(digit):
    return float(digit)


def convertFaceVectorToJSON(face_vector):
    json_string = json.dumps({"faceVector": face_vector }, default=converter)

    return json_string.encode(encoding='utf_8')


class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/calculate-face-vector':
            try:
                content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
                data = self.rfile.read(content_length)  # <--- Gets the data itself
                face_vector = calc_embs(data)
                face_vector_json = convertFaceVectorToJSON(face_vector)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(face_vector_json)
            except Exception as e:
                self.send_error(500, str(e))
                self.end_headers()

