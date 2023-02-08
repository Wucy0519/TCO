# Targeted confidence optimization (TCO)

This project is implementation for: ***”Mix-supervised Learning via Targeted Confidence Optimization for Image Classification“***. In this work, inspired by human learning, we propose mix-supervised learning and give a learning paradigm to solve this challenge. Based on this paradigm, we design the targeted confidence optimization (TCO) to perform mix-supervised learning, which can effectively eliminate the feature scattering due to pseudo-label, meanwhile preserving the role of pseudo-labeling in promoting model convergence.

## Project Directory

```
.
|--- main.py					#Train your model on this script
|--- requirements.txt				#Environment requirements for the code to run
|--- utils					   
|    |--- wcy_method.py				#The script of TCO method
|    └--- weight_init.py			#Initialize linear layer parameters
|--- model
|    |--- model.py				#Calling the model
|    └--- TransFG.py				#Defining and Initializing VIT
|--- dataset
|    |--- dataset.py				#Calling the dataset
|    |--- read_imagenet.py			#Processing the dataset: Imagenet-1K
|    |--- cub.py				#Processing the dataset: CUB_200_2011
|    └--- VOC2007.py				#Processing the dataset: VOC2007
|--- Save					#Save model parameters (Need to create)
└--- pretrained					#The pretrained models are here (Need to create)
```

## Installation

pip install required packages

```shell
#Python == 3.8.5
pip install -r requirements.txt
```

go to code folder

```shell
cd /tco
```

## Training

Train multiple models on various datasets:

```python
#main.py

#Add the dataset name you want to train to this list
dataset_list = ["cifar100","VOCdevkit","CUB"]
#Add the model name you want to train to this list
model_list = ["ViT","DenseNet169","Resnet50"]
```

- If you want to train vit (the default is TransFG), add string "ViT" to "model_list", download the pretrained model, add it to "pretrained" fold and name it "vit.npz"

Change batch size for each model:

```python
#main.py

bs_list = {"DenseNet169":128,"ViT":32,"Resnet50":128}
```

Add the dataset paths :

```python
#Your dataset path. Such as : /home/wcy/DataSet
path = '/XXX/XXX/DataSet'
```

- The format of the dataset file is as follows:


```
.
└--- Dataset
    |--- cifar100				#CIFAR100 Dataset		
    |	└--- ...		
    |--- CUB					#CUB_200_2011 Dataset	
    |	└--- ...	
	|--- ImageNet-1k			#ImageNet-1K Dataset		
    |	└--- ...		
    └--- Vocdevkit				#VOC Dataset	
    	|--- VOC2007
    	└--- ...		
```

Set the  parameter of TCO:

```python
#main.py

#Random drop optimizer parameter
p_drop = 0.8

#The confidence threshold
s_l = 0.7

#The switch of TCO method. 
#If you want use TCO to train your model, turn it into True
tco_switch = True 
```

Initialize the TCO:
```python
#main.py

if tco_switch:
    Tco_mode = Tco(net = Wcynet,		#You model
                   test_loader = test_loader,	#Test set
                   drop_p = p_drop,
                   s_l = s_l,
                   gamma = gamma,		#scheduler's parameter
                   steps = step_size,		#scheduler's parameter
                   lr = base_lr,
                   weight_decay = weight_decay,
                   momentum=momentum,
                   solver_name="doubleSGD") 		
else:
    print("Not use TCO method in your training")
```

- doubleSGD: Random Drop Optimizer transformed form SGD.

Run the script ***main.py***:
```shell
python main.py
```
