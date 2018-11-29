import Algorithmia
from uuid import uuid4
from os import remove
client = Algorithmia.client()

def get_json(filename):
    return client.file(filename).getJson()

def get_file(filename):
    return client.file(filename).getFile().name

def upload_image(local_path, remote_prefix):
    img_name = "{}.png".format((str(uuid4())))
    full_remote_path = "{}/{}".format(remote_prefix, img_name)
    client.file(full_remote_path).putFile(local_path)
    remove(local_path)
    return full_remote_path


class AlgorithmError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)