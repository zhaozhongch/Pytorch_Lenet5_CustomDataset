This is a tutorial level pytorch implementation of Lenet5.  
The difference of this code between others is that this implementation tries to use PNG file to train/validate lenet5 instead of using one line code such as
```
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)
```
to load the dataset.  
The goal is to make people know how to load their own dataset with pytorch.  

##Environment
pytroch: 1.4  
CUDA: 10.1  
CUDNN: 7  
python: 3.7.4  
As this repo doesn't include high-level library, any pytorch>1.0 and python>3.0 should work.  

#Use
Clone this repo and download the MNIST digital dataset
```
git clone https://github.com/zhaozhongch/Pytorch_Lenet5_CustomDataset.git
cd Pytorch_Lenet5_CustomDataset
#clone the dataset
git clone https://github.com/myleott/mnist_png.git
cd mnist_png
tar -xvf mnist_png.tar.gz
```
  
To run the trainning network with only CPU, run 
```
python lenet5_train_cpu.py
```
A `lenet5.pth` model will be generated at the current location. To verify the accuracy
```
python lenet5_test_cpu.py
```
  
Run lenet5 with GPU
```
python lenet5_train_gpu.py
```
Verify the accuracy
```
python lenet5_test_gpu.py
```

##Some comments about the code
1:The original [lenet5 paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) is designed for input image 32 by 32, but the dataset in fact in png file in fact is 28*28. Not a big deal, just set the input array size 28 by 28.   
2:There are many duplicate parts in the code such as reading dataset and the network structure. As a tutorial level code, I just try to make every file can run independently