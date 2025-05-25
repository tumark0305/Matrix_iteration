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
from itertools import product

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

class method:
    def __init__(self , _data:header_info):
        self.data = _data
        self.cross_list = [[] for _ in range(len(self.data.net_data))]

    def find_crossing(self):
        self.cross_list = [[] for _ in range(len(self.data.net_data))]
        for _a in range(0 , len(self.data.net_data)):
            for _b in range(_a+1 ,len(self.data.net_data)):
                _cross_coord = method.cross_at(self.data.net_data[_a] , self.data.net_data[_b])
                self.cross_list[_a] += _cross_coord
                self.cross_list[_b] += _cross_coord
        return None
    def cross_at(_pina_data , _pinb_data):
        _neta_area = set([tuple(p) for p in _pina_data.line_coordinate] +  [tuple(p) for p in _pina_data.pin_coordinate])
        _netb_area = set([tuple(p) for p in _pinb_data.line_coordinate] +  [tuple(p) for p in _pinb_data.pin_coordinate])
        _output = list(_neta_area & _netb_area)
        return _output
    def direct_connection(_pin_location):
        output = [[],[]] #|^ and _|
        _LD_point = np.array([min(_pin_location.T[0]) , min(_pin_location.T[1])])
        _RT_point = np.array([max(_pin_location.T[0]) , max(_pin_location.T[1])])
        _Dhor = [[_x , _LD_point[1]] for _x in range(_LD_point[0] + 1, _RT_point[0])]
        _Rver = [[_RT_point[0] , _y] for _y in range(_LD_point[1] + 1 , _RT_point[1])]
        _Thor = [[_x , _RT_point[1]] for _x in range(_LD_point[0] + 1, _RT_point[0])]
        _Lver = [[_LD_point[0] , _y] for _y in range(_LD_point[1] + 1 , _RT_point[1])]
        _LT_point = np.array([_LD_point[0] , _RT_point[1]])
        _RD_point = np.array([_RT_point[0] , _LD_point[1]])
        if _pin_location[0][0] > _pin_location[1][0]:
            _Dhor.reverse()
            _Thor.reverse()
        if _pin_location[0][1] > _pin_location[1][1]:
            _Rver.reverse()
            _Lver.reverse()

        if len(_Dhor) == 0 or len(_Rver) == 0:
            output[0] = _Lver + _Thor
            output[1] = _Dhor + _Rver
        else:
            if np.linalg.norm(_LD_point - _pin_location[0]) == 0:
                output[0] = _Lver + [_LT_point.tolist()] + _Thor
                output[1] = _Dhor + [_RD_point.tolist()] + _Rver
            elif np.linalg.norm(_LT_point - _pin_location[0]) == 0:
                output[0] = _Lver + [_LD_point.tolist()] + _Dhor
                output[1] = _Thor + [_RT_point.tolist()] + _Rver
            elif np.linalg.norm(_RD_point - _pin_location[0]) == 0:
                output[0] = _Rver + [_RT_point.tolist()] + _Thor
                output[1] = _Dhor + [_LD_point.tolist()] + _Lver
            elif np.linalg.norm(_RT_point - _pin_location[0]) == 0:
                output[0] = _Rver + [_RD_point.tolist()] + _Dhor
                output[1] = _Thor + [_LT_point.tolist()] + _Lver

        return output

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
    def loss_cal_cross(self):
        _output = 0
        for _i,_pin in enumerate(self.data.net_data):
            for _cmp_pin in self.data.net_data[_i+1:]:
                for _patha in _pin.line_strip:
                    for _pathb in _cmp_pin.line_strip:
                        _eq_start = _patha[0][0] == _pathb[0][0] and _patha[0][1] == _pathb[0][1]
                        _eq_end = _patha[1][0] == _pathb[1][0] and _patha[1][1] == _pathb[1][1]
                        if _eq_end:#corss
                            _output += 1
        return _output
    def loss_cal_bend(self):
        _output = 0
        for _pin in self.data.net_data:
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
                    _output += 1
                _current_direction = _next_direction
        return _output
    def loss_cal_length(self):
        _output = 0
        for _x in self.data.net_data:
            _output += len(_x.line_strip)
        return _output
    def loss(self):
        self.convert_toline()
        _wire_length = self.loss_cal_length()
        _cross = self.loss_cal_cross()
        _bend = self.loss_cal_bend()
        _total_loss = _wire_length * self.data.propagation_loss + _cross * self.data.crossing_loss + _bend * self.data.bending_loss
        self.data_pack.append([_total_loss,_wire_length,_cross,_bend])
        return None
    def convert_toline(self):
        for _pin in self.data.net_data:
            _pin.connected = False
            _pin.line_strip = []
            _current_coordinate = _pin.pin_coordinate[0]#start assume
            _end_coordinate = _pin.pin_coordinate[1]#end
            _connect_in = _pin.line_coordinate.copy()
            _x_isclose = _current_coordinate[0] -1 <= _connect_in[0][0] and _connect_in[0][0] <= _current_coordinate[0] +1 
            _y_isclose = _current_coordinate[1] -1 <= _connect_in[0][1] and _connect_in[0][1] <= _current_coordinate[1] +1 
            if not (_x_isclose and _y_isclose):
                _current_coordinate = _pin.pin_coordinate[1]#start change
                _end_coordinate = _pin.pin_coordinate[0]#end
            _connect_in.append(_end_coordinate)
            for _next_coordinate in _connect_in:
                _line = np.array([_current_coordinate,_next_coordinate])
                _pin.line_strip.append(_line)
                _current_coordinate = _next_coordinate
            _pin.connected = True
        return None
    def direct_connect(self):
        if len(self.data.net_data) < 25:
            self.data.net_data = PIC.best_direct_connect_in_list(self.data.net_data)
        else:
            _data_buffer = PIC.from_smallest_direct_connect_in_list(self.data.net_data)
            _crossing_table = [0 for _ in range(len(_data_buffer))]
            for _a in range(0,len(_data_buffer)):
                for _b in range(_a +1,len(_data_buffer)):
                    _crossing_count = len(method.cross_at(_data_buffer[_a] , _data_buffer[_b]))
                    _crossing_table[_a] += _crossing_count
                    _crossing_table[_b] += _crossing_count
            _reroute_list = []
            _noroute_list = []
            for _it in range(len(_crossing_table)):
                if _crossing_table[_it] == 0:
                    _noroute_list.append(_data_buffer[_it])
                else:
                    _reroute_list.append(_data_buffer[_it])
            _finish_reroute = PIC.best_direct_connect_in_list(_reroute_list)
            self.data.net_data = _finish_reroute + _noroute_list
        self.matrix = self.convert_tomatrix()
        self.all_matrix.append(self.matrix)
        return None

    @staticmethod
    def best_direct_connect_in_list(_pin_list:list[net_info])->list[net_info]:
        _output = copy.deepcopy(_pin_list)
        all_options = [method.direct_connection(net.pin_coordinate) for net in _pin_list]
        all_selectors = list(product([0, 1], repeat=len(all_options)))
    
        best_cross_count = float('inf')
        best_routing = None

        for selector in all_selectors:
            for i, route_idx in enumerate(selector):
                _pin_list[i].line_coordinate = all_options[i][route_idx]
        
            total_cross = 0
            for i in range(len(_pin_list)):
                total_cross += sum(
                    len(method.cross_at(_pin_list[i], _pin_list[j]))
                    for j in range(i+1, len(_pin_list))
                )

            if total_cross < best_cross_count:
                best_cross_count = total_cross
                best_routing = [net.line_coordinate.copy() for net in _pin_list]

        # 最佳路徑寫回
        for i in range(len(_pin_list)):
            _output[i].line_coordinate = best_routing[i]
        return _output

    @staticmethod
    def from_smallest_direct_connect_in_list(pin_list: list[net_info]) -> list[net_info]:
        output = []
        distances = [np.linalg.norm(net.pin_coordinate[0] - net.pin_coordinate[1]) for net in pin_list]
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

        for idx in sorted_indices:
            original_net = pin_list[idx]
            route_options = method.direct_connection(original_net.pin_coordinate)
            cross_counts = []

            for option in route_options:
                test_net = copy.deepcopy(original_net)
                test_net.line_coordinate = option
                cross_count = sum(len(method.cross_at(test_net, placed)) for placed in output)
                cross_counts.append(cross_count)

            best_route = route_options[np.argmin(cross_counts)]
            selected_net = copy.deepcopy(original_net)
            selected_net.line_coordinate = best_route
            output.append(selected_net)

        return output

            
    def forward(self):
        _method = method(self.data)
        _method.find_crossing()
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
