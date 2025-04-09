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

LabelBase.register(name='chinese', fn_regular='C:\\Windows\\Fonts\\msjh.ttc')
class EDA_method:
    def __init__(self):
        self.col, self.row = 10,15
        self.matrix = None
        self.all_matrix = []
        self.data_pack = []
        self.block_count = 10
        self.nonsquare_count = 2
        self.coordinate_list = []
        self.size_list = []
        self.HPWL_list = []
        self.feasible_list = []
    def load_random_matrix(self):
        _raw_mat = np.random.randint(0, 100, size=(self.col, self.row), dtype=np.uint8)
        _top_indices = np.unravel_index(np.argsort(_raw_mat, axis=None)[-self.block_count:], _raw_mat.shape)
        _block_size_x = np.random.randint(1, 4, size=(len(_top_indices[0])), dtype=np.uint8)
        _block_size_y = _block_size_x.copy()[:len(_top_indices[0])-self.nonsquare_count]
        _block_size_y = np.concatenate((_block_size_y, np.random.randint(1, 4, size=(self.nonsquare_count), dtype=np.uint8)))
        
        [self.coordinate_list.append([_top_indices[0][_idx],_top_indices[1][_idx]]) for _idx in range(self.block_count)]
        [self.size_list.append([_block_size_x[_idx],_block_size_y[_idx]]) for _idx in range(self.block_count)]
        self.coordinate_list = np.array(self.coordinate_list, dtype=np.uint8)
        self.size_list = np.array(self.size_list, dtype=np.uint8)
        self.matrix = self.convert_tomatrix()
        self.all_matrix.append(self.matrix)
        
        return None
    def convert_tomatrix(self):
        _output = np.zeros((self.col, self.row), dtype=np.uint8)
        for _y in range(len(_output)):
            for _x in range(len(_output[_y])):
                for _block_idx,_coordinate in enumerate(self.coordinate_list):
                    if abs(_coordinate[0] - _x) <=self.size_list[_block_idx][0] and abs(_coordinate[1] - _y) <self.size_list[_block_idx][1]:
                        _output[_y][_x] +=1
        return _output
    def loss(self):
        def feasible()->bool:
            _all_x = []
            _all_y = []
            for _idx in range(self.block_count):
                _all_x.append(self.coordinate_list[_idx][0] + self.size_list[_idx][0])
                _all_x.append(self.coordinate_list[_idx][0] - self.size_list[_idx][0])
                _all_y.append(self.coordinate_list[_idx][1] + self.size_list[_idx][1])
                _all_y.append(self.coordinate_list[_idx][1] - self.size_list[_idx][1])
            _L = min(_all_x) >= 0
            _R = max(_all_x) < self.row
            _B = min(_all_y) >=0
            _T = max(_all_y) < self.col
            _overlap = not np.any(self.matrix >= 2)
                
            return all([_L,_R,_T,_B,_overlap])
        def grade():
            _max_value = np.max(self.matrix)
            _max_mask = (self.matrix == _max_value)
            _labeled_array, _num_features = ndimage.label(_max_mask)
            _max_centroids = ndimage.center_of_mass(_max_mask, _labeled_array, range(1, _num_features + 1))
            _zero_mask = (self.matrix == 0)
            _labeled_array, _num_features = ndimage.label(_zero_mask)
            _zero_centroids = ndimage.center_of_mass(_zero_mask, _labeled_array, range(1, _num_features + 1))
            if len(_zero_centroids) <= 1:
                pass#single
            else:
                pass #bi
            return None
        _sep_coordinate = np.transpose(self.coordinate_list)
        _HPWL = (max(_sep_coordinate[0]) - min(_sep_coordinate[0]) + max(_sep_coordinate[1]) - min(_sep_coordinate[1]))/2
        _feasible = feasible()
        self.HPWL_list.append(_HPWL)
        self.feasible_list.append(_feasible)
        grade()
        self.data_pack.append([_HPWL,_feasible])
        return None
    def forward(self):
        self.loss()
        for i in range(5):
            self.coordinate_list += 1
            self.matrix = self.convert_tomatrix()
            self.loss()
            self.all_matrix.append(self.matrix)
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
            _output = BoxLayout(orientation='horizontal',
                                  size_hint=(None, None),
                                  height=40)
            _output.add_widget(self.HPWL_label)
            _output.add_widget(self.feasible_label)
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
        
        
        root = BoxLayout(orientation='vertical',
                         spacing=10,
                         padding=10,
                         size_hint=(None, None))
        root.add_widget(self.grid)
        for _button in __all_button():
            root.add_widget(_button)
        root.add_widget(__step_controller())
        root.add_widget(__label_display0())

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
    

    def prev_step(self, instance):
        if self.step_count > 0:
            self.step_count -= 1
            self.step_label.text = f"步數 = {self.step_count}/{len(self.mat_list)}"
            self.HPWL_label.text = f"HPWL = {self.data_pack[self.step_count][0]:.4f}"
            self.feasible_label.text = f"可行 = {self.data_pack[self.step_count][1]}"
            self.matrix = self.mat_list[self.step_count]
            self.__refresh_matrix()
        return None
    def next_step(self, instance):
        if self.step_count < len(self.mat_list)-1:
            self.step_count +=1
            self.step_label.text = f"步數 = {self.step_count}/{len(self.mat_list)}"
            self.HPWL_label.text = f"HPWL = {self.data_pack[self.step_count][0]:.4f}"
            self.feasible_label.text = f"可行 = {self.data_pack[self.step_count][1]}"
            self.matrix = self.mat_list[self.step_count]
            self.__refresh_matrix()
        else:
            self.step_label.text = f"步數 = {self.step_count}/{len(self.mat_list)}"
            self.HPWL_label.text = f"HPWL = {self.data_pack[self.step_count][0]:.4f}"
            self.feasible_label.text = f"可行 = {self.data_pack[self.step_count][1]}"
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
