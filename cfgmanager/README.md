| [**Installation**](#installation)
| [**Main functions**](#main-functions)
| [**hydra guide**](#hydra)

# What is **cfgmanager**?
**cfgmanager** is a [*hydra*](https://hydra.cc/docs/intro/) config file manager, it helps to **save** and **find** the *hydra* config files.

## Contents
* [Installation](#installation)
* [Main functions](#main-function)
* [hydra](#hydra)

## Installation
1. pull this directory to your computer.
2. cd to this directory (which contains `setup.py`).
3. use the following command to install (editable).
```bash
pip install -e .
```

## Main functions  
**cfgmanager** contains two main functions, which are `cfgmanager.save` and `cfgmanager.find`.

### 1. save
`cfgmanager.save` is used to save *hydra* cfg into **config.yaml** file to your data path.
- ***import***
```python
from cfgmanager import save
```
- ***use***
```python
save(cfg, save_path)
```
- This example shows how to save `cfg` into `save_path`.
- In this minmal example, `cfg` is an object of `omegaconf.DictConfig`, which is used in *hydra* while loading configs from **.yaml** file. 
- `save_path` is the directory string you want to save the **config.yaml** file in, then you can find the saved **config.yaml** in your `save_path`. 
- The **config.yaml** file you saved contains all the parameters in `cfg`, and it also contains some "auto_save" information like time, git, etc.
- For more usage please refer to the examples and docstrings.

### 2. find
`cfgmanager.find` is used to find the saved **config.yaml** file in your data path.
- ***import***
```python
from cfgmanager import find
```
- ***use***
```python
search_dict = {'num': 4, 'T': 10000}
find(directory, search_dict)
```
- This example shows how to find all the **config.yaml** files in `directory` which match the keys and valus in `search_dict`.
- `cfgmanager.find` will use `os.walk` to find all the matched **config.yaml** files in every subdirectory of `directory`.
- For more usage please refer to the examples and docstrings.

## hydra
[*hydra*](https://hydra.cc/docs/intro/) is a python package which simplifies the procedure of loading configuration parameters for python programs.

This is a minmal tutorial of using **cfgmanager** while using [*hydra*](https://hydra.cc/docs/intro/). 
### 1. run & save
- The `config` file for *hydra* to load
```yaml
# ./conf/config.yaml
parameter_a: 1
parameter_b: 2.5
db:
  parameter_c: string
  parameter_d: True
```
- The `my_app.py` python scripy to run
```python
# ./my_app.py
import hydra
from cfgmanager import save
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(cfg.parameter_a)
    print(cfg.parameter_b)
    print(cfg.db.parameter_c)
    print(cfg.db.parameter_d)

    save_path = "./data/"
    save(cfg, save_path)

if __name__ == "__main__":
    my_app()
```
- Run `my_app.py` and override some values in the loaded config from the command line if you want
```bash
$ python my_app.py parameter_a=2 db.parameter_d=False
save config to:
File "/home/user/test/data/config.yaml"
```
- Then you can find the saved overridden **config.yaml** file
```yaml
# ./data/config.yaml
parameter_a: 2
parameter_b: 2.5
db:
  parameter_c: string
  parameter_d: False
auto_save_time: 2024-11-08_12:00:00
auto_save_path:
  call_path: /home/user/test/
  cfg_path: /home/user/test/data/config.yaml
auto_save_git:
  branch: main
  hash: 6dad1508f2367c663009d0463f936cb100138654
  message: blablabla
auto_save_cfgmanager_version: '1.0'
```
### 2. find
- The `find_data.py` 
```python
# ./find_data.py
from cfgmanager import find
find('./data/', {parameter_b: 2.5})
```
- Run `find_data.py`
```bash
$ python find_data.py
Find 1 matched config files:
File "/home/user/test/data/config.yaml"

Brief matched configs:
   parameter_a  parameter_b  db_parameter_c  db_parameter_d
0            2          2.5          string           False
``` 