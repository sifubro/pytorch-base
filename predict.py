import torch
import json
from torchvision.io import read_image
from gunicorn.app.base import BaseApplication
from model import SimpleNet  # Import your PyTorch model
from preprocessing import test_transform

'''
Instructions
--------------

docker build -t my-pytorch-app .
docker run -p 8000:8000 -v my-pytorch-app
curl -X POST -H "Content-Type: application/json" -d "{\"image_path\": \"cat.3880.jpg\"}" http://localhost:8000

'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./1.pt"

# Load your PyTorch model
model = SimpleNet().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = 2*(image/255.0) - 1 # between -1 and +1
        return image



def predict(image_path):
    global model
    global device

    # read fname and apply transform
    img = test_transform(read_image(image_path))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    #return output.argmax(dim=1).item()
    return {"prediction" : output.item()}


def app(environ, start_response):
    if environ['REQUEST_METHOD'] != 'POST':
        start_response('405 Method Not Allowed', [('Content-Type', 'text/plain')])
        return [b'Method Not Allowed']

    try:
        content_length = int(environ.get('CONTENT_LENGTH', '0'))
    except ValueError:
        content_length = 0

    request_body = environ['wsgi.input'].read(content_length)
    request_data = json.loads(request_body.decode())

    if 'image_path' not in request_data:
        start_response('400 Bad Request', [('Content-Type', 'text/plain')])
        return [b'Bad Request']

    image_path = request_data['image_path']
    result = predict(image_path)

    start_response('200 OK', [('Content-Type', 'application/json')])
    return [json.dumps(result).encode()]


class PredictionApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load(self):
        return self.application


if __name__ == '__main__':
    options = {
        'bind': '0.0.0.0:8000',
        'workers': 1
    }


    PredictionApplication(app, options).run()