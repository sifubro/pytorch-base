import torchvision.transforms as transforms

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = 2*(image/255.0) - 1 # between -1 and +1
        return image
    


# Define image transformation
test_transform = transforms.Compose([
    Normalize(),
    transforms.Resize(256),
    transforms.CenterCrop(224)
    ])


