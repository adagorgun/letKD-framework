# letKD - Framework
Official PyTorch implementation of our BMVC 2023 paper **Knowledge Distillation Layer that Lets the Student Decide**, an approach for knowledge distillation.

We adopt the framework implemented by [QuEST](https://github.com/valeoai/quest).

## **Preparation**

### **Pre-requisites**
* Python >= 3.7
* PyTorch >= 1.3.1
* CUDA 10.0 or higher

faiss-gpu, sacred, imageio, matplotlib, numpy, opencv_python, Pillow, scikit_image, scikit_learn, skimage, tensorboardX, tqdm

#### Config files
Config files named as CIFAR100/<*model*>/<*TeacherNetwork_StudentNetwork*>/<*letKD-1*>or <*letKD-2*>. 

<*model*>:resnetv1, WRN and cross_model.

## **Instructions**

**(1)** Run download_cifar100_teacher.py to download all the teacher models used in the paper for CIFAR100. 

**(2)** Run main_classification.py for example train as follows:

#### Train student network:
```bash
# Train letKD-1 with teacher-student: ResNet56-ResNet20
python main_classification.py with config=CIFAR100.resnetv1.RN56_RN20.letKD-1 
# output logs are stored at ./experiments/CIFAR100/resnetv1/RN56_RN20/letKD-1 

# Train letKD-2 with teacher-student: ResNet56-ResNet20
python main_classification.py with config=CIFAR100.resnetv1.RN56_RN20.letKD-2 
# output logs are stored at ./experiments/CIFAR100/resnetv1/RN56_RN20/letKD-2
```

**(3)** Run main_classification.py for example evaluation as follows:

#### Evaluate student network:
```bash
# Evaluate letKD-1 with teacher-student: ResNet56-ResNet20
python main_classification.py with config=CIFAR100.resnetv1.RN56_RN20.letKD-1 evaluate=True
# output logs are stored at ./experiments/CIFAR100/resnetv1/RN56_RN20/letKD-1 

# Evaluate letKD-2 with teacher-student: ResNet56-ResNet20
python main_classification.py with config=CIFAR100.resnetv1.RN56_RN20.letKD-2 evaluate=True
# output logs are stored at ./experiments/CIFAR100/resnetv1/RN56_RN20/letKD-2
```