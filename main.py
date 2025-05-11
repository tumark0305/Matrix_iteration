from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ListProperty
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
import copy,os
import numpy as np
from scipy import ndimage
from tqdm import trange

LabelBase.register(name='chinese', fn_regular='C:\\Windows\\Fonts\\msjh.ttc')

class net_info:
    def __init__(self,_line_text):
        self.id = int(_line_text.split(' ')[0])
        self.pin_coordinate = np.array([[int(_line_text.split(' ')[1]),int(_line_text.split(' ')[2])],[int(_line_text.split(' ')[3]),int(_line_text.split(' ')[4])]])
        self.line_coordinate = []
        self.line_strip = []
        self.connected = False
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
        self.all_matrix = []
        self.data_pack = []
        self.matrix = None
        self.input_path = f"{os.getcwd()}/pa2_resources/{input_file}"
        self.input_text = PIC.__read_file(self.input_path)
        self.data = header_info(self.input_text)
        self.col = self.data.grid[0]
        self.row = self.data.grid[1]
    @staticmethod
    def __read_file(_path):
        _f = open(_path,'r')
        _data = _f.read()
        _f.close()
        return _data
    def convert_tomatrix(self):
        _output = np.zeros((self.col, self.row), dtype=np.uint8)
        for _y in range(len(_output)):
            for _x in range(len(_output[_y])):
                for _connection in self.data.net_data:
                    for _pin_coord in _connection.pin_coordinate:
                        if _x == _pin_coord[0] and _y == _pin_coord[1]:
                            _output[_y][_x] += 50
                    for _line_coord in _connection.line_coordinate:
                        if _x == _line_coord[0] and _y == _line_coord[1]:
                            _output[_y][_x] +=1
        return _output
    def loss(self):
        _total_loss = 0
        _wire_length = 0
        _cross = 0
        _bend = 0
        for _i,_pin in enumerate(self.data.net_data):
            for _cmp_pin in self.data.net_data[_i+1:]:
                for _patha in _pin.line_strip:
                    for _pathb in _cmp_pin.line_strip:
                        _eq_start = _patha[0][0] == _pathb[0][0] and _patha[0][1] == _pathb[0][1]
                        _eq_end = _patha[1][0] == _pathb[1][0] and _patha[1][1] == _pathb[1][1]
                        if _eq_start and _eq_end:#overlap
                            _cross += 1
                            pass
                        elif _eq_end:#corss
                            _cross += 1
            _wire_length += len(_pin.line_strip)
            _direction_overview = []
            _current_direction = "N"
            for _path in _pin.line_strip:
                _next_direction = "N"
                if _path[0][0] == _path[1][0] :
                    _next_direction = "Y"
                elif _path[0][1] == _path[1][1] :
                    _next_direction = "X"
                else:
                    print("not a valid line")
                _direction_overview.append(_next_direction)
                if _current_direction != "N" and _current_direction != _next_direction:
                    _bend += 1
                _current_direction = _next_direction
        _total_loss = _wire_length * self.data.propagation_loss + _cross * self.data.crossing_loss + _bend * self.data.bending_loss
        self.data_pack.append([_total_loss,_wire_length,_cross,_bend])
        return None
    def direct_connect(self):
        for _pin in self.data.net_data:
            _pin.connected = False
            _pin.line_coordinate = []
            _pin.line_strip = []
            _current_coordinate = np.array([min(_pin.pin_coordinate.T[0]),min(_pin.pin_coordinate.T[1])])
            for _wire_x in range(min(_pin.pin_coordinate.T[0]),max(_pin.pin_coordinate.T[0])+1):
                _wire_y = min(_pin.pin_coordinate.T[1])
                _next_coordinate = np.array([_wire_x,_wire_y])
                _eq_pin0 = _next_coordinate[0] == _pin.pin_coordinate[0][0] and _next_coordinate[1] == _pin.pin_coordinate[0][1]
                _eq_pin1 = _next_coordinate[0] == _pin.pin_coordinate[1][0] and _next_coordinate[1] == _pin.pin_coordinate[1][1]
                if  not(_eq_pin0 or _eq_pin1):
                    _pin.line_coordinate.append(_next_coordinate)
                if not(_next_coordinate[0] == _current_coordinate[0] and _next_coordinate[1] == _current_coordinate[1]):
                    _pin.line_strip.append(np.array([_current_coordinate,_next_coordinate]))
                _current_coordinate = _next_coordinate
            for _wire_y in range(min(_pin.pin_coordinate.T[1]),max(_pin.pin_coordinate.T[1])+1):
                _wire_x = max(_pin.pin_coordinate.T[0])
                _next_coordinate = np.array([_wire_x,_wire_y])
                _eq_pin0 = _next_coordinate[0] == _pin.pin_coordinate[0][0] and _next_coordinate[1] == _pin.pin_coordinate[0][1]
                _eq_pin1 = _next_coordinate[0] == _pin.pin_coordinate[1][0] and _next_coordinate[1] == _pin.pin_coordinate[1][1]
                if  not(_eq_pin0 or _eq_pin1):
                    _pin.line_coordinate.append(_next_coordinate)
                if not(_next_coordinate[0] == _current_coordinate[0] and _next_coordinate[1] == _current_coordinate[1]):
                    _pin.line_strip.append(np.array([_current_coordinate,_next_coordinate]))
                _current_coordinate = _next_coordinate
            _pin.connected = True
        return None
            
    def forward(self):
        self.matrix = self.convert_tomatrix()
        self.all_matrix.append(self.matrix)
        return None

class MatrixIterationVisualize(App):
    def __init__(self , matrix_list, data_pack , **kwargs):
        super().__init__(**kwargs)
        self.mat_list = matrix_list
        self.data_pack = data_pack
        self.matrix = self.mat_list[0]
        self.col, self.row = self.matrix.shape
        self.cell_width = 80   # 每格大概 6 字元寬60
        self.cell_height = 40  # 每格高度
        self.font_size = 24
        self.step_count = 0

    def build(self):
        def __all_button():
            reset_btn = Button(text='重設',
                               font_name='chinese',
                               size_hint=(None, None),
                               size=(self.cell_width * self.row, 40),
                               font_size=self.font_size)
            reset_btn.bind(on_press=self.reset_matrix)

            random_btn = Button(text='隨機產生',
                               font_name='chinese',
                               size_hint=(None, None),
                               size=(self.cell_width * self.row, 40),
                               font_size=self.font_size)
            random_btn.bind(on_press=self.random_matrix)

            
            return [reset_btn, random_btn]
        def __step_controller():
            prev_btn = Button(text='上一步',
                              font_name='chinese',
                              size_hint=(None, None),
                              size=(self.cell_width * self.row / 3, 40),
                              font_size=self.font_size)
            prev_btn.bind(on_press=self.prev_step)

            self.step_label = Label(text = f"步數 = {self.step_count+1}/{len(self.mat_list)}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))

            next_btn = Button(text='下一步',
                              font_name='chinese',
                              size_hint=(None, None),
                              size=(self.cell_width * self.row / 3, 40),
                              font_size=self.font_size)
            next_btn.bind(on_press=self.next_step)
            _output = BoxLayout(orientation='horizontal',
                                  size_hint=(None, None),
                                  height=40)
            _output.add_widget(prev_btn)
            _output.add_widget(self.step_label)
            _output.add_widget(next_btn)
            return _output
        def __label_display0():
            self.HPWL_label = Label(text = f"光損耗 = {self.data_pack[self.step_count][0]}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))

            self.feasible_label = Label(text = f"總線長 = {self.data_pack[self.step_count][1]}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))
            self.global_vector_label = Label(text = f"交叉 = {self.data_pack[self.step_count][2]}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))
            _output = BoxLayout(orientation='horizontal',
                                  size_hint=(None, None),
                                  height=40)
            _output.add_widget(self.HPWL_label)
            _output.add_widget(self.feasible_label)
            _output.add_widget(self.global_vector_label)
            return _output
        self.grid = GridLayout(cols=self.row,
                               spacing=2,
                               size_hint=(None, None),orientation='lr-bt')
        self.grid.bind(minimum_width=self.grid.setter('width'),
                       minimum_height=self.grid.setter('height'))
        self.labels = []

        for row in self.matrix:
            for value in row:
                lbl = Label(text=f"{value:02d}",
                            font_size=self.font_size,
                            size_hint=(None, None))
                lbl.texture_update()
                lbl.size = (self.cell_width, self.cell_height)
                self.labels.append(lbl)
                self.grid.add_widget(lbl)
        self.__refresh_matrix()
        self.coord_label = Label(text = f"折角 = {self.data_pack[self.step_count][3]}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row, 40))
        
        root = BoxLayout(orientation='vertical',
                         spacing=10,
                         padding=10,
                         size_hint=(None, None))
        root.add_widget(self.grid)
        for _button in __all_button():
            root.add_widget(_button)
        root.add_widget(__step_controller())
        root.add_widget(__label_display0())
        root.add_widget(self.coord_label)

        root.bind(minimum_width=root.setter('width'),
                  minimum_height=root.setter('height'))

        # 根據矩陣大小設定初始視窗大小
        total_width = self.cell_width * self.row + 40
        total_height = self.cell_height * self.col + 40 + 50
        Window.size = (total_width, total_height)

        # ➤ 讓整體置中
        anchor = AnchorLayout()
        anchor.add_widget(root)
        return anchor
    
    def refresh_data(self):
        self.step_label.text = f"步數 = {self.step_count+1}/{len(self.mat_list)}"
        self.HPWL_label.text = f"光損耗 = {self.data_pack[self.step_count][0]}"
        self.feasible_label.text = f"總線長 = {self.data_pack[self.step_count][1]}"
        self.global_vector_label.text = f"交叉 = {self.data_pack[self.step_count][2]}"
        self.coord_label.text = f"折角 = {self.data_pack[self.step_count][3]}"
        self.matrix = self.mat_list[self.step_count]
        self.__refresh_matrix()
        return None
    def prev_step(self, instance):
        if self.step_count > 0:
            self.step_count -= 1
            self.refresh_data()
        return None
    def next_step(self, instance):
        if self.step_count < len(self.mat_list)-1:
            self.step_count +=1
            self.refresh_data()
        return None
    def reset_matrix(self, instance):
        self.matrix.fill(0.0)  
        self.__refresh_matrix()
        return None
    def __refresh_matrix(self):
        for _i,_value in enumerate(self.matrix.reshape(-1)):
            self.labels[_i].text = f"{_value:02d}"
            if _value == 0:
                self.labels[_i].color = (1, 1, 1, 1) 
            elif _value == 1:
                self.labels[_i].color = (0, 1, 0, 1)  
            elif _value == 2:
                self.labels[_i].color = (0, 0, 1, 1)  
            elif _value >=50:
                self.labels[_i].color = (1, 0, 0, 1)  
            else:
                self.labels[_i].color = (1, 1, 0, 1)  
        return None

    def random_matrix(self, instance):
        self.matrix = np.random.randint(0, 100, size=(self.col, self.row))
        self.__refresh_matrix()
        return None



if __name__ == '__main__':
    _project = PIC("pic5x5.in")
    _project.direct_connect()
    _project.loss()
    _project.forward()
    _project.loss()
    MatrixIterationVisualize(_project.all_matrix,_project.data_pack).run()
