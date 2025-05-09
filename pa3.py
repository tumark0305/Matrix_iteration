import os
from tqdm import tqdm

class net_info:
    def __init__(self,_line_text):
        self.id = int(_line_text.split(' ')[0])
        self.coordinate0 = [int(_line_text.split(' ')[1]),int(_line_text.split(' ')[2])]
        self.coordinate1 = [int(_line_text.split(' ')[3]),int(_line_text.split(' ')[4])]
class header_info:
    def __init__(self,_input_text):
        self.grid = [int(_input_text.split('\n')[0].split(' ')[1]),int(_input_text.split('\n')[0].split(' ')[2])]
        self.propagation_loss = int(_input_text.split('\n')[1].split(' ')[2])
        self.crossing_loss = int(_input_text.split('\n')[2].split(' ')[2])
        self.bending_loss = int(_input_text.split('\n')[3].split(' ')[2])
        self.num_net = int(_input_text.split('\n')[4].split(' ')[2])
        self.net_data = [net_info(_x) for _x in _input_text.split('\n')[5:] if _x!=""]

class PIC:
    def __init__(self,input_file:str):
        self.input_path = f"{os.getcwd()}/pa2_resources/{input_file}"
        self.input_text = PIC.__read_file(self.input_path)
        self.data = header_info(self.input_text)
    @staticmethod
    def __read_file(_path):
        _f = open(_path,'r')
        _data = _f.read()
        _f.close()
        return _data
    def run(self):
        return None

if __name__ == '__main__':
    pa = PIC("pic5x5.in")
    pa.run()



