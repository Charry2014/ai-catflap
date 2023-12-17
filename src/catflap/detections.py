'''
Classes for wrapping the detection results from TFLite

'''

class BoundingRect():
    def __init__(self, origin_x: int, origin_y:int, width:int, height:int ) -> None:
        '''The same as used by tflite but defined here for ease of use'''
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"BoundingRect(origin_x={self.origin_x}, origin_y={self.origin_y}, width={self.width}, height={self.height})"

class TFLDetection():
    def __init__(self, label:str, index: int, score:float, box:BoundingRect) -> None:
        self.label = label
        self.index = index
        self.score = score
        self.box = box

    def __iter__(self):
        yield self.label
        yield self.score
        yield self.box

    def __str__(self) -> str:
        return f"{self.index}-{self.label}"
        
    @property
    def detail(self) ->str:
        return f"{self} {self.score:.2f} {self.box}"

class TFLDetections():
    def __init__(self, file_name:str) -> None:
        '''Container class for a list of detections as returned by tflite for a single image
        When a camera is used as a source it is identified by its integer ID'''
        if isinstance(file_name, int):
            self.file_name = f"Camera {file_name}"
        else:
            self.file_name = file_name
        self._list = []
    
    def add(self, name:str, index:int, score:float, box:BoundingRect) -> None:
        self._list.append(TFLDetection(name, index, score, box))

    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        for i in self._list:
            yield i

    def __str__(self):
        return self.file_name

    def __next__(self):
       pass
    
    def __getitem__(self, i):
        return self._list[i]