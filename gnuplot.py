import os




if __name__ == '__main__':
    _f = open(f"{os.getcwd()}\\output.gp","r")
    _data = _f.read()
    _f.close()
    _data_list = _data.split('\n')
    _output = [_x for _x in _data_list if "center" not in _x]
    _text = "\n".join(_output)
    _f = open(f"{os.getcwd()}\\new_output.gp","w")
    _f.write(_text)
    _f.close()





