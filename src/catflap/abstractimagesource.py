from abc import ABC, abstractmethod
from array import array

class AbstractImageSource(ABC):
    def __init__(self, **kwargs):
        '''Turn the provided arguments into attributes and validate
        that a source was specified'''
        for key, value in kwargs.items():
            setattr(self, key, value)
        assert getattr(self, 'source')
        self._isopen = False

    @property
    def isopen(self) -> bool:
        return self._isopen
    
    @staticmethod
    @abstractmethod
    def can_supply_images(source: str) -> bool:
        '''Returns True if this class can supply images from the source'''
        return False

    @abstractmethod
    def open_image_source(self) -> int:
        '''Opens an image source from the provided arguments 
        Returns 0 for success, or anything else for an error'''
        return 0

    @abstractmethod
    def close_image_source(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        return 0

    @abstractmethod
    def get_image(self) -> array:
        '''Gets the next image from an image source
        Returns the image, or None when there are no more'''
        return None
