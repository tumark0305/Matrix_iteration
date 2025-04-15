from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ListProperty
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
import copy
import numpy as np
from scipy import ndimage
from tqdm import trange

LabelBase.register(name='chinese', fn_regular='C:\\Windows\\Fonts\\msjh.ttc')

class block_info:
    reach_counter = 0
    col, row = 8,15
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.reach_counter += 1
        instance.reach_counter = cls.reach_counter
        return instance
    def __init__(self,_coordinate = np.zeros(2,dtype=np.int32),_size = np.zeros(2,dtype=np.int32)):
        self.coordinate = _coordinate.copy()#x,y
        self.size = _size.copy()#x,y
        self.step = np.zeros(2,dtype=np.int32)
        self.history_coordinate = [_coordinate.copy()]
        self.global_vector = np.zeros(2,dtype=np.int32)
        self.new_coordinate = _coordinate.copy()
        self.tag = f"{self.reach_counter}"
        self.sublock = []
    def clip_coordinate(self):
        _output = self.coordinate.copy()
        if _output[0] +self.size[0] > self.row:
            _output[0] = self.row-self.size[0]
        elif _output[0] < 0:
            _output[0] = 0
        if _output[1] +self.size[1]> self.col:
            _output[1] = self.col-self.size[1]
        elif _output[1] < 0:
            _output[1] = 0
        return _output
    def move(self):
        self.history_coordinate.append(self.coordinate.copy())
        self.coordinate += self.step
        self.coordinate = self.clip_coordinate()
        self.global_vector = self.history_coordinate[0] - self.coordinate
        if len(self.sublock) >0:
            _all_x = []
            _all_y = []
            for _sublock in self.sublock:
                _all_x.append(_sublock.coordinate[0])
                _all_y.append(_sublock.coordinate[1])
            _real_coord = np.array([min(_all_x),min(_all_y)],dtype=np.int32)
            _offset = self.coordinate - _real_coord
            for _sublock in self.sublock:
                _sublock.coordinate += _offset
        return None
    def unprotect_move(self):
        self.history_coordinate.append(self.coordinate.copy())
        self.coordinate += self.step
        for _sublock in self.sublock:
            _sublock.coordinate += self.step
        self.global_vector = self.history_coordinate[0] - self.coordinate
        return None
    def teleport(self):
        self.history_coordinate.append(self.coordinate.copy())
        self.coordinate = self.new_coordinate.copy()
        self.coordinate = self.clip_coordinate()
        _difference =  self.history_coordinate[-1] - self.coordinate
        for _sublock in self.sublock:
            _sublock.coordinate += _difference
        self.global_vector = self.history_coordinate[0] - self.coordinate
        return None
    def unprotect_teleport(self):
        self.history_coordinate.append(self.coordinate.copy())
        _difference =  (self.new_coordinate - self.coordinate).copy()
        self.coordinate = self.new_coordinate.copy()
        for _sublock in self.sublock:
            _sublock.coordinate += _difference
        self.global_vector = self.history_coordinate[0] - self.coordinate
        return None
    def cal_from_sublock(self):
        self.tag = "combined"
        _all_x = []
        _all_y = []
        _all_size = np.zeros(2,dtype=np.int32)
        _global_vector = np.zeros(2,dtype=np.int32)
        for _block in self.sublock:
            _all_x.append(_block.coordinate[0])
            _all_y.append(_block.coordinate[1])
            _all_size += _block.size
            _global_vector += _block.history_coordinate[0] - _block.coordinate
        self.coordinate = np.array([min(_all_x),min(_all_y)],dtype=np.int32)
        self.size = _all_size
        self.global_vector = _global_vector
        return None

class EDA_method:
    max_block_size = [8,1]
    abacus_alpha = 1
    def __init__(self):
        self.col, self.row = block_info.col , block_info.row
        self.matrix = None
        self.all_matrix = []
        self.data_pack = []
        self.block_count = 20
        self.HPWL_list = []
        self.feasible_list = []
        self.all_global_vector_list = []
        self.block_list = []
        self.placed_block_list = []
    def load_random_matrix(self):
        _raw_mat = np.random.randint(0, 100, size=(self.col, self.row), dtype=np.uint8)
        _top_indices = np.unravel_index(np.argsort(_raw_mat, axis=None)[-self.block_count:], _raw_mat.shape)
        if self.max_block_size[0] > 1:
            _block_size_x = np.random.randint(1, self.max_block_size[0], size=(len(_top_indices[0])), dtype=np.uint8)
        else:
            _block_size_x = np.ones(len(_top_indices[0]), dtype=np.uint8)
        if self.max_block_size[1] > 1:
            _block_size_y = np.random.randint(1, self.max_block_size[1], size=(len(_top_indices[1])), dtype=np.uint8)
        else:
            _block_size_y = np.ones(len(_top_indices[1]), dtype=np.uint8)
        [self.block_list.append(block_info(np.array([_top_indices[1][_idx],_top_indices[0][_idx]],dtype=np.int32),np.array([_block_size_x[_idx],_block_size_y[_idx]],dtype=np.int32))) for _idx in range(self.block_count)]
        return None
    def convert_tomatrix(self):
        _output = np.zeros((self.col, self.row), dtype=np.uint8)
        for _y in range(len(_output)):
            for _x in range(len(_output[_y])):
                for _block in self.block_list:
                    if 0<= _x - _block.coordinate[0] < _block.size[0] and 0<=_y-_block.coordinate[1]<_block.size[1]:
                        _output[_y][_x] +=1
        return _output
    def loss(self,_method:str):
        assert _method in ["spring" , "abacus"]
        def feasible()->bool:
            _all_x = []
            _all_y = []
            for _block in self.block_list:
                _all_x.append(_block.coordinate[0] + _block.size[0])
                _all_x.append(_block.coordinate[0])
                _all_y.append(_block.coordinate[1] + _block.size[1])
                _all_y.append(_block.coordinate[1])
            _L = min(_all_x) >= 0
            _R = max(_all_x) <= self.row
            _B = min(_all_y) >=0
            _T = max(_all_y) <= self.col
            _overlap = not np.any(self.matrix >= 2)
                
            return all([_L,_R,_T,_B,_overlap])
        def HPWL()->float:
            _all_x = []
            _all_y = []
            for _block in self.block_list:
                _all_x.append(_block.coordinate[0])
                _all_y.append(_block.coordinate[1])
            _output = (max(_all_x) - min(_all_x) + max(_all_y) - min(_all_y))/2
            return _output
        def global_vector()->float:
            _result = 0
            for _block in self.block_list:
                _result += np.linalg.norm(_block.global_vector) / len(self.block_list)
            return _result
        def spring_method():
            _sum_force = [np.array([0,0],dtype=np.int32) for _ in range(len(self.block_list))]
            for _a in range(len(self.block_list)):
                for _b in range(_a+1,len(self.block_list)):
                    _overlap = overlap(self.block_list[_a],self.block_list[_b])
                    if _overlap[0]>0 and _overlap[1]>0:
                        _overlap = np.array(_overlap,dtype = np.int32)
                        _spring_force = np.clip(np.ceil(_overlap/2),0,None).astype(np.int32)
                        _min_force = _spring_force
                        if np.random.random() < 0.9:
                            if _spring_force[0] > _spring_force[1]:
                                _min_force[0] = 0
                            elif _spring_force[0] < _spring_force[1]:
                                _min_force[1] = 0
                        else:
                            if _spring_force[0] > _spring_force[1]:
                                _min_force[1] = 0
                            elif _spring_force[0] < _spring_force[1]:
                                _min_force[0] = 0
                            else:
                                _min_force[1] = 0
                        if np.random.random() < 0.7:
                            _sum_force[_a] += _min_force
                        else:
                            _sum_force[_b] -= _min_force
            return np.array(_sum_force,dtype=np.int32)
        def overlap(_blocka,_blockb)->list[int]:
            _bordera = [[_blocka.coordinate[0],_blocka.coordinate[0]+_blocka.size[0]],[_blocka.coordinate[1],_blocka.coordinate[1]+_blocka.size[1]]]
            _borderb = [[_blockb.coordinate[0],_blockb.coordinate[0]+_blockb.size[0]],[_blockb.coordinate[1],_blockb.coordinate[1]+_blockb.size[1]]]
            _overlap = [min(_bordera[0][1],_borderb[0][1]) - max(_bordera[0][0],_borderb[0][0]),min(_bordera[1][1],_borderb[1][1]) - max(_bordera[1][0],_borderb[1][0])]
            return _overlap
        def abacus_method():
            _alpha = 1
            def cal_cost(_input_block:block_info , _if_atrow:int)->float:
                def combine_block(_block_new:block_info , _block_placed:block_info)->block_info:
                    _combine_block = block_info()
                    _block_new.new_coordinate = _block_placed.coordinate + _block_placed.size#
                    _block_new.new_coordinate[1] = _block_new.coordinate[1]
                    _block_new.unprotect_teleport()
                    if len(_block_new.sublock) == 0:
                        _combine_block.sublock.append(_block_new)
                    else:
                        _combine_block.sublock.extend(_block_new.sublock)
                    if len(_block_placed.sublock) == 0:
                        _combine_block.sublock.append(_block_placed)
                    else:
                        _combine_block.sublock.extend(_block_placed.sublock)
                    _combine_block.cal_from_sublock()
                    _combine_block.size[1] = 1
                    _subcoord = [_sub.coordinate for _sub in _combine_block.sublock]
                    return _combine_block
                def cal_complex_loss(_now_block:block_info )->float:
                    def unpack(_placed_mirror:list[block_info])->list[block_info]:
                        _output_blocks = []
                        for _combined_block in _placed_mirror:
                            if len(_combined_block.sublock) == 0:
                                _output_blocks.append(_combined_block)
                            else:
                                _output_blocks.extend(_combined_block.sublock)
                        return copy.deepcopy(_output_blocks)
                    _save_tag = _now_block.tag
                    _new_block = copy.deepcopy(_now_block)
                    _new_block.tag = "current"
                    _effected_blocks = []
                    _changed = True
                    while _changed:
                        _changed = False
                        for _placed_without_current_block in _placed_mirror0.copy():#mirror1 目標group是否碰到已存在group，有就收入目標group
                            _overlap = overlap(_new_block , _placed_without_current_block)
                            if _overlap[0]>=0 and _overlap[1]>0:
                                _effected_blocks.append(_placed_without_current_block)
                                _placed_mirror0.remove(_placed_without_current_block)
                                _new_block = combine_block(_new_block , _placed_without_current_block)
                                _sun_vector = 0
                                for _sublock in _new_block.sublock:
                                    _sun_vector += _sublock.history_coordinate[0][0] - _sublock.coordinate [0]
                                _new_block.step[0] = int(_sun_vector / len(_new_block.sublock))
                                _new_block.step[1] = 0
                                _new_block.move()
                                _changed = True
                                break
                    _placed_mirror0.append(_new_block)
                    _afterD = 0
                    _DL = 0
                    for _sublock in _new_block.sublock:
                        if _sublock.tag == "current":
                            _DL += np.linalg.norm(_sublock.history_coordinate[0] - _sublock.coordinate)
                            _sublock.tag = _save_tag
                        else:
                            _afterD += np.linalg.norm(_sublock.history_coordinate[0] - _sublock.coordinate)
                    _beforeD = 0
                    for _block in _effected_blocks:#_effected_blocks從mirror0來的不會有群組
                        _beforeD += np.linalg.norm(_block.history_coordinate[0] - _block.coordinate)
                    _cost = _alpha * _DL + _afterD - _beforeD

                    _placed_condition = unpack(_placed_mirror0)
                    _oversize = [_combined.size[0] > block_info.row for _combined in _placed_mirror0]
                    _overlap = []
                    
                    if any(_oversize):
                        _cost = np.inf
                    for _a in range(len(_placed_condition)):
                        for _b in range(_a+1 , len(_placed_condition)):
                            _overlap = overlap(_placed_condition[_a] , _placed_condition[_b])
                            if _overlap[0]>0 and _overlap[1]>0:
                                _cost = np.inf
                    return _cost , _placed_condition

                _placed_mirror0 = copy.deepcopy(_placed)#選項計算不能影響現實
                _now_block = copy.deepcopy(_input_block)
                _now_block.new_coordinate[1] = _if_atrow
                _now_block.teleport()
                _output = None
                for _placed_block in _placed_mirror0.copy():#mirror0 目標是否碰到已存在
                    _overlap = overlap(_now_block , _placed_block)
                    if _overlap[0]>0 and _overlap[1]>0:
                        _output,_placed_condition = cal_complex_loss(_now_block)
                        break
                if _output is None:
                    _output = _alpha * np.linalg.norm(_now_block.history_coordinate[0] - _now_block.coordinate)
                    _placed_mirror0.append(_now_block)
                    _placed_condition = _placed_mirror0
                return _output , _placed_condition
            _placed = []
            _block_list = [copy.deepcopy(self.block_list[_x]) for _x in range(len(self.block_list))]
            _all_x = np.array([_block.coordinate[0] for _block in self.block_list])
            _sorted = np.argsort(_all_x)
            for _block_idx in _sorted:
                _all_cost = []
                _all_condition = []
                for _option_row in range(block_info.col):
                    _cost,_condition = cal_cost(_block_list[_block_idx],_option_row)
                    _all_cost.append(_cost)
                    _all_condition.append(_condition)
                _sorted_cost_idx = np.argsort(_all_cost)
                _placed = _all_condition[_sorted_cost_idx[0]]
            _output = []
            for _block_orig in _block_list:
                for _block_data in _placed:
                    if _block_orig.tag == _block_data.tag:
                        _output.append(_block_data.coordinate)
                        break
            return _output
        _feasible = feasible()
        _HPWL = HPWL()
        _orig_coord = [_block.coordinate.copy() for _block in self.block_list]
        _global_vector = global_vector()
        self.HPWL_list.append(_HPWL)
        self.feasible_list.append(_feasible)
        self.all_global_vector_list.append(_global_vector)
        self.data_pack.append([_HPWL,_feasible,_global_vector,_orig_coord])
        if _method == "spring":
            _sum_force = spring_method()
            for _x in range(len(self.block_list)):
                self.block_list[_x].step = _sum_force[_x]
        elif _method == "abacus":
            _orig_coord = [_block.coordinate for _block in self.block_list]
            _all_coord = abacus_method()
            for _x in range(len(self.block_list)):
                self.block_list[_x].new_coordinate = _all_coord[_x]
        return None
    def forward(self):
        _block_list_copy = [copy.deepcopy(_block) for _block in self.block_list]
        for _ in range(10):
            [_block.teleport() for _block in self.block_list]
            self.matrix = self.convert_tomatrix()
            self.all_matrix.append(self.matrix.copy())
            self.loss("abacus")

        self.block_list = _block_list_copy
        for _ in range(1000):
            [_block.move() for _block in self.block_list]
            self.matrix = self.convert_tomatrix()
            self.all_matrix.append(self.matrix.copy())
            self.loss("spring")
            if self.feasible_list[-1] :
                break
        return None

class MatrixIterationVisualize(App):
    def __init__(self , matrix_list, data_pack , **kwargs):
        super().__init__(**kwargs)
        self.mat_list = matrix_list
        self.data_pack = data_pack
        self.matrix = self.mat_list[0]
        self.col, self.row = self.matrix.shape
        self.cell_width = 60   # 每格大概 6 字元寬
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

            self.step_label = Label(text = f"步數 = {self.step_count}/{len(self.mat_list)}",
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
            self.HPWL_label = Label(text = f"HPWL = {self.data_pack[self.step_count][0]:.4f}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))

            self.feasible_label = Label(text = f"可行 = {self.data_pack[self.step_count][1]}",
                                font_name='chinese',
                                font_size=self.font_size,
                                size_hint=(None, None),
                                size=(self.cell_width * self.row/3, 40))
            self.global_vector_label = Label(text = f"全局距離 = {self.data_pack[self.step_count][2]:.2f}",
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
                               size_hint=(None, None))
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
        _coord_text = ",".join([f"({_x:02d},{_y:02d})" for _x,_y in self.data_pack[self.step_count][3]])
        self.coord_label = Label(text = f"方塊座標 = {_coord_text}",
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
        self.step_label.text = f"步數 = {self.step_count}/{len(self.mat_list)}"
        self.HPWL_label.text = f"HPWL = {self.data_pack[self.step_count][0]:.4f}"
        self.feasible_label.text = f"可行 = {self.data_pack[self.step_count][1]}"
        self.global_vector_label.text = f"全局距離 = {self.data_pack[self.step_count][2]:.2f}"
        _coord_text = ",".join([f"({_x:02d},{_y:02d})" for _x,_y in self.data_pack[self.step_count][3]])
        self.coord_label.text = f"方塊座標 = {_coord_text}"
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
            else:
                self.labels[_i].color = (1, 0, 0, 1)  
        return None

    def random_matrix(self, instance):
        self.matrix = np.random.randint(0, 100, size=(self.col, self.row))
        self.__refresh_matrix()
        return None



if __name__ == '__main__':
    _project = EDA_method()
    _project.load_random_matrix()
    _project.forward()
    MatrixIterationVisualize(_project.all_matrix,_project.data_pack).run()
