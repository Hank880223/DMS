# DMS
## Cross compile ncnn with i.MX8M Plus / i.MX8 QuadMax for aarch64
    $ git clone https://github.com/Tencent/ncnn.git
    $ cd ncnn
    $ git submodule update --init
    $ mkdir build && cd build
    $ . /opt/bsp-5.4.70-2.3.3/environment-setup-aarch64-poky-linux 
### You can refer to it from here [BSP](https://github.com/Hank880223/ncnn-sort-vehicle/blob/main/doc/BSP.md)
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

## Run Folder

```
└─build folder
    ├─model
    │   ├─det1.bin
    │   ├─det1.param
    │   ├─det2.bin
    │   ├─det2.param
    │   ├─det3.bin
    │   ├─det3.param
    │   ├─face.bin
    │   ├─face.param
    │   ├─face1.bin
    │   ├─face1.param
    │   ├─landmark106.bin
    │   ├─landmark106.param
    │   ├─mobilefacenet.bin
    │   ├─mobilefacenet.param
    │   ├─mobilenetv2-yolov3-128x128-0.5.bin
    │   ├─mobilenetv2-yolov3-128x128-0.5.param
    │   ├─yolo-fastest.bin
    │   └─yolo-fastest.param
    ├─User-information
    │   ├─Han-Wei
    │   │   ├─sample0.jpg
    │   │   └─sample1.jpg
    │   ├─Qing-Long
    │   │   ├─sample0.jpg
    │   │   └─sample1.jpg
    │   └─RUI-LI
    │       ├─sample0.jpg
    │       └─sample1.jpg
    ├─icon
    │   ├─head0.jpg
    │   ├─head1.jpg
    │   ├─phone0.jpg
    │   ├─phone1.jpg
    │   ├─sleep0.jpg
    │   ├─sleep1.jpg
    │   ├─smoke0.jpg
    │   └─smoke1.jpg
    └─DMS
```  
