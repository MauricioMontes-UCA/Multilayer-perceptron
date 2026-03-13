from numpy.typing import NDArray
import numpy as np

class Loader:

    @staticmethod
    def load_mnist_images(filename: str) -> NDArray:
        with open(filename, 'rb') as f:
            magic: int = int.from_bytes(f.read(4), 'big')
            num_images: int = int.from_bytes(f.read(4), 'big')
            rows: int = int.from_bytes(f.read(4), 'big')
            cols: int = int.from_bytes(f.read(4), 'big')

            images: NDArray = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)

            return images
        
    @staticmethod
    def load_mnist_labels(filename: str) -> NDArray:
        with open(filename, 'rb') as f:
            magic: int = int.from_bytes(f.read(4), 'big')
            num_labels: int = int.from_bytes(f.read(4), 'big')

            labels: NDArray = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        

class Transformer:
    
    @staticmethod
    # El nivel de iluminación se basa en un valor entre 0 (negro) y 255 (blanco)
    # El proceso devuelve una versión normalizada entre 0 y 1.
    def transform_digits(digits_set: NDArray) -> NDArray:
        digits_set = digits_set.astype(np.float32) / 255.0
        digits_set = digits_set.reshape(-1, 28*28)
        
        return digits_set

    @staticmethod
    # One-hot encoding consiste en codificar varias categorías en un vector de componentes binarios
    # Solo puede haber un componente que sea 1, pues cada componente es un booleano de si el dato
    # pertenece a esa clase.
    def transform_labels(labels_set: NDArray) -> NDArray:
        num_clases: int = 10

        return np.eye(num_clases)[labels_set]