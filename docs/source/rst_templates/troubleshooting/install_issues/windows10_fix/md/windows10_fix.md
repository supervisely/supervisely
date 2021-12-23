If you are experiencing problems with supervisely_lib import in your python interpreter, 
this is probably because of the `shapely` module that can not be installed correctly on Windows 10

You can solve this problem by downloading shapely binary from [unofficial site](<https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>)
and entering a few commands in your terminal

This will uninstall incorrect version of shapely
```
>>> pip uninstall shapely 
```
Download the correct version of shapely that will suit your machine. 

`cp38` in the file name stands for CPython 3.8.x version

In our case we use version for Python 3.8.5 and Windows 10 x64:
```
>>> pip install Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
```
