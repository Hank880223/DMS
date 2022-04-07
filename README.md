# DMS
## Cross compile ncnn with i.MX8M Plus / i.MX8 QuadMax for aarch64
    $ git clone https://github.com/Tencent/ncnn.git
    $ cd ncnn
    $ git submodule update --init
    $ mkdir build && cd build
    $ . /opt/bsp-5.4.70-2.3.3/environment-setup-aarch64-poky-linux
    $ cmake ..
    $ make -j`nproc` && make install
    
move install/include and lib folder to DMS folder
```
└─DMS
    ├─include
    │  └─ncnn
    ├─lib
    │  └─libncnn.a
    └─src
```    
    
## Build DMS
    $ mkdir build && cd build
    $ . /opt/bsp-5.4.70-2.3.3/environment-setup-aarch64-poky-linux
    $ cmake ..
    $ make -j`nproc`

## Run

```
└─DMS
    ├─models
    │   ├─yolov3-tiny_obj_opt.param
    │   └─yolov3-tiny_obj_opt.bin
    └─tracking
```  
