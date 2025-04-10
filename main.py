from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ListProperty
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
import numpy as np
from scipy import ndimage
from tqdm import trange

LabelBase.register(name='chinese', fn_regular='C:\\Windows\\Fonts\\msjh.ttc')
class EDA_method:
    max_block_size = 8
    def __init__(self):
        self.col, self.row = 15,20
        self.matrix = None
        self.all_matrix = []
        self.data_pack = []
        self.block_count = 20
        self.nonsquare_count = 2
        self.coordinate_list = []
        self.size_list = []
        self.HPWL_list = []
        self.feasible_list = []
        self.all_coordinate_list = []
        self.all_global_vector_list = []
    def load_random_matrix(self):
        _raw_mat = np.random.randint(0, 100, size=(self.col, self.row), dtype=np.uint8)
        _top_indices = np.unravel_index(np.argsort(_raw_mat, axis=None)[-self.block_count:], _raw_mat.shape)
        _block_size_x = np.random.randint(1, self.max_block_size, size=(len(_top_indices[0])), dtype=np.uint8)
        _block_size_y = _block_size_x.copy()[:len(_top_indices[0])-self.nonsquare_count]
        _block_size_y = np.concatenate((_block_size_y, np.random.randint(1, self.max_block_size, size=(self.nonsquare_count), dtype=np.uint8)))
        
        [self.coordinate_list.append([_top_indices[0][_idx],_top_indices[1][_idx]]) for _idx in range(self.block_count)]
        [self.size_list.append([_block_size_x[_idx],_block_size_y[_idx]]) for _idx in range(self.block_count)]
        self.coordinate_list = np.array(self.coordinate_list, dtype=np.uint8)
        self.size_list = np.array(self.size_list, dtype=np.uint8)
        self.all_coordinate_list.append(self.coordinate_list.copy())
        
        return None
    def convert_tomatrix(self):
        _output = np.zeros((self.col, self.row), dtype=np.uint8)
        for _y in range(len(_output)):
            for _x in range(len(_output[_y])):
                for _block_idx,_coordinate in enumerate(self.coordinate_list):
                    if 0 <= _x - _coordinate[0]  <self.size_list[_block_idx][0] and 0 <= _y - _coordinate[1] <self.size_list[_block_idx][1]:
                        _output[_y][_x] +=1
        return _output
    def loss(self):
        def calculate_block_centers():
            centers = []
            for idx in range(len(self.coordinate_list)):
                x, y = self.coordinate_list[idx]
                width, height = self.size_list[idx]
                center_x = int(x + width / 2)
                center_y = int(y + height / 2)
                centers.append([center_x, center_y])
            return np.array(centers)
        def feasible()->bool:
            _all_x = []
            _all_y = []
            for _idx in range(self.block_count):
                _all_x.append(self.coordinate_list[_idx][0] + self.size_list[_idx][0])
                _all_x.append(self.coordinate_list[_idx][0])
                _all_y.append(self.coordinate_list[_idx][1] + self.size_list[_idx][1])
                _all_y.append(self.coordinate_list[_idx][1])
            _L = min(_all_x) >= 0
            _R = max(_all_x) < self.row
            _B = min(_all_y) >=0
            _T = max(_all_y) < self.col
            _overlap = not np.any(self.matrix >= 2)
                
            return all([_L,_R,_T,_B,_overlap])
        def field_grade():
            def uni_charge():
                _soft_e = 1e-3
                _soft_pw = 1.5
                _max_step = 3

                _pos_charge = np.zeros((self.col, self.row,2), dtype=np.float32)
                for _pos in _max_centroids:
                    _pos_array = np.array(_pos)
                    for _y in range(self.col):
                        for _x in range(self.row):
                            _diff = np.array([_x, _y]) - _pos_array
                            _r2 = np.sum(_diff ** 2)
                            if _r2 != 0:
                                _r = np.sqrt(_r2)
                                _pos_charge[_y][_x] += (_diff / (_r + _soft_e)) * (1 / (_r**_soft_pw + _soft_e)) * _max_value
                _result = np.ceil(np.abs(_pos_charge)) * np.sign(_pos_charge)
                _delta = np.clip(_result, -_max_step, _max_step)
                return _delta.astype(np.int32)
            def bi_charge():
                _output = np.zeros((self.col, self.row, 2), dtype=np.float32)
                _pos_charge = _output.copy()
                _neg_charge = _output.copy()
    
                for _pos in _max_centroids:
                    _pos_array = np.array(_pos)
                    for _y in range(self.col):
                        for _x in range(self.row):
                            _diff = np.array([_x, _y]) - _pos_array
                            _r2 = np.sum(_diff ** 2)
                            if _r2 != 0:
                                _r = np.sqrt(_r2)
                                _pos_charge[_y][_x] += (_diff / _r) * (1 / _r) * _max_value

                for _neg in _zero_centroids:
                    _neg_array = np.array(_neg)
                    for _y in range(self.col):
                        for _x in range(self.row):
                            _diff = _neg_array - np.array([_x, _y])
                            _r2 = np.sum(_diff ** 2)
                            if _r2 != 0:
                                _r = np.sqrt(_r2)
                                _neg_charge[_y][_x] += (_diff / _r) * (1 / _r) * _max_value

                _output = _pos_charge + _neg_charge
                _result = np.ceil(np.abs(_output)) * np.sign(_output)
                return _result.astype(np.int32) 

            _center_coord = calculate_block_centers()
            _max_value = np.max(self.matrix)
            _max_mask = (self.matrix == _max_value)
            _labeled_array, _num_features = ndimage.label(_max_mask)
            _max_centroids = ndimage.center_of_mass(_max_mask, _labeled_array, range(1, _num_features + 1))
            #_max_centroids = np.mean(np.argwhere(self.matrix >= 2), axis=0)
            _zero_mask = (self.matrix == 0)
            _labeled_array, _num_features = ndimage.label(_zero_mask)
            _zero_centroids = ndimage.center_of_mass(_zero_mask, _labeled_array, range(1, _num_features + 1))
            if len(_zero_centroids) <= 1:
                _field_matrix = uni_charge() 
            else:
                _field_matrix = uni_charge() 
                #_field_matrix = bi_charge()
            _vectors = []
            for _idx in range(self.block_count):
                _vet = [0,0]
                _coord = [_center_coord[_idx][0],_center_coord[_idx][1]]
                if _center_coord[_idx][0] + self.size_list[_idx][0] >= self.row:
                    _vet[0] = self.row - _center_coord[_idx][0] - self.size_list[_idx][0]
                    _coord[0] = self.row-1
                if _center_coord[_idx][1] + self.size_list[_idx][1] >= self.col:
                    _vet[1] = self.col - _center_coord[_idx][1] - self.size_list[_idx][1]
                    _coord[1] = self.col-1
                if _center_coord[_idx][0] <0:
                    _vet[0] = 0
                    _coord[0] = 0
                if _center_coord[_idx][1] <0:
                    _vet[1] = 0
                    _coord[1] = 0
                _vectors.append(_field_matrix[_coord[1], _coord[0]] + np.array(_vet))
            _new_coord = (self.coordinate_list + np.array(_vectors))
            _new_coord_clip = np.clip(_new_coord, 0,None)
            self.coordinate_list = np.array(_new_coord_clip, dtype=np.uint8)
            self.all_coordinate_list.append(self.coordinate_list.copy())
            return None
        def fast_method():
            _sep_coordinate = np.transpose(self.coordinate_list)
            _order = np.argsort(_sep_coordinate[0])
            _e0 = 1
            _sub_sum0 = 0
            for _idx0 in range(1,len(_order)):
                _sub_sum1 = 0
                for _idx1 in range(1,_idx0):
                    _sub_sum1 += self.size_list[_order[_idx1]][0]
                _sub_sum0 += _e0 * (_sep_coordinate[0][_order[_idx0]] - _sub_sum1)
            _sub_sum0 += _e0 * _sep_coordinate[0][_order[0]]
            _sub_sum3 = _e0 * len(_order)
            _x0 = int(_sub_sum0 / _sub_sum3)
            if _x0 <0:
                _x0 = 0
            _coordinate_list = self.coordinate_list.copy()
            _xn = _x0
            for _order_idx in _order:
                _cell = [_xn,_sep_coordinate[1][_order_idx]]
                _coordinate_list[_order_idx] = _cell
                _xn += self.size_list[_order_idx][0]
            self.coordinate_list = np.array(_coordinate_list, dtype=np.uint8)
            self.all_coordinate_list.append(self.coordinate_list.copy())

            return None
        def HPWL()->float:
            _sep_coordinate = np.transpose(self.coordinate_list)
            _output = (max(_sep_coordinate[0]) - min(_sep_coordinate[0]) + max(_sep_coordinate[1]) - min(_sep_coordinate[1]))/2
            return _output
        def global_vector()->float:
            _v1 = self.all_coordinate_list[0]
            _v2 = self.coordinate_list
            _diff = _v1.astype(int) - _v2.astype(int)
            _result = np.abs(_diff).sum()
            return _result
        def spring_method():
            _coordinate = self.coordinate_list.astype(np.int32)
            _size = self.size_list.astype(np.int32)
            def cal_block_force():
                _sum_force = [np.array([0,0],dtype=np.int32) for _ in range(self.block_count)]
                for _blocka in range(self.block_count):
                    _bordera = [[_coordinate[_blocka][0],_coordinate[_blocka][0]+_size[_blocka][0]],[_coordinate[_blocka][1],_coordinate[_blocka][1]+_size[_blocka][1]]]
                    for _blockb in range(_blocka+1,self.block_count):
                        _borderb = [[_coordinate[_blockb][0],_coordinate[_blockb][0]+_size[_blockb][0]],[_coordinate[_blockb][1],_coordinate[_blockb][1]+_size[_blockb][1]]]
                        _overlap = [min(_bordera[0][1],_borderb[0][1]) - max(_bordera[0][0],_borderb[0][0]),min(_bordera[1][1],_borderb[1][1]) - max(_bordera[1][0],_borderb[1][0])]
                        if _overlap[0]>0 and _overlap[1]>0:
                            _overlap = np.array(_overlap,dtype = np.int32)
                            _spring_force = np.clip(np.ceil(_overlap/2),0,None).astype(np.int32)
                            _sum_force[_blocka] += _spring_force
                            _sum_force[_blockb] -= _spring_force
                return np.array(_sum_force,dtype=np.int32)
            _sum_force = cal_block_force()
            _new_coordinate_raw = (_sum_force + _coordinate).T
            _new_coordinate_clipB = np.array([np.clip(_new_coordinate_raw[0],0,self.row),np.clip(_new_coordinate_raw[1],0,self.col)],dtype=np.int32)+_size.T
            _new_coordinate_clip = np.array([np.clip(_new_coordinate_clipB[0],0,self.row-1),np.clip(_new_coordinate_clipB[1],0,self.col-1)],dtype=np.int32)-_size.T
            self.coordinate_list = np.array(_new_coordinate_clip.T, dtype=np.uint8)
            self.all_coordinate_list.append(self.coordinate_list.copy())
            return None
        
        _feasible = feasible()
        _HPWL = HPWL()
        _orig_coord = self.coordinate_list.copy()
        _global_vector = global_vector()
        self.HPWL_list.append(_HPWL)
        self.feasible_list.append(_feasible)
        self.all_global_vector_list.append(_global_vector)
        #field_grade()
        spring_method()
        #fast_method()
        self.data_pack.append([_HPWL,_feasible,_global_vector,_orig_coord])
        return None
    def forward(self):
        for i in trange(60):
            self.matrix = self.convert_tomatrix()
            self.all_matrix.append(self.matrix.copy())
            self.loss()
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
            self.global_vector_label = Label(text = f"全局距離 = {self.data_pack[self.step_count][2]}",
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
        self.global_vector_label.text = f"全局距離 = {self.data_pack[self.step_count][2]}"
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
