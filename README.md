# **Attention Network with Optional Activation Function for Unsupervised Video Anomaly Detection**  

This repository contains the **PyTorch implementation** of our research paper published in the **ETRI Journal**. You can read the full paper here: [ETRI Journal](https://onlinelibrary.wiley.com/doi/10.4218/etrij.2024-0115).  

## **📌 Overview**  
This project implements a **Attention-based Autoencoder Network with Optional Activation Function** designed for unsupervised video anomaly detection. The method leverages PyTorch for training and evaluation, ensuring efficient learning and robust performance.  

## **🚀 Getting Started**  

### **1. Prerequisites**  
Ensure you have the following dependencies installed before running the code:  

- **Python** 3.8+  
- **PyTorch** 1.7.1  
- **Torchvision** 0.8.2  

### **2. Install Dependencies**  
To install the required packages, run:  

```sh
pip install -r requirements.txt
```  

### **3. Prepare Datasets**  
Download and place the dataset in the `datasets/` folder. Ensure the data is structured correctly before running the model.  

### **4. Configure Hyperparameters**  
Modify hyperparameters such as **learning rate, number of epochs, batch size**, etc., in the `.yaml` file inside the `configs/` directory.  

---

## **🔥 Training the Model**  
To train the model, run:  

```sh
python train.py
```  

## **🎯 Testing the Model**  
To evaluate the model on test data, run:  

```sh
python test.py
```  

---

## **📂 Project Structure**  
```
Image-Analysis-ETRI/
│── config/           # Configuration files (configs.yaml)
│── datasets/          # Folder to store datasets
│── models/            # Model architecture definitions
│── utils/             # Helper functions and utilities
│── train.py           # Script for training the model
│── test.py            # Script for testing the model
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
```  

## **🛠 Troubleshooting & Tips**  
- Ensure you have the correct **Python** and **PyTorch** versions installed.  
- If dependencies are missing, manually install them using `pip install <package-name>`.  
- Adjust hyperparameters in `configs.yaml` for improved model performance.  

---

## **📜 Citation**  
If you find this work useful, please cite our paper:  

```
@article{rakhmonov2024aonet,
  title={AONet: Attention network with optional activation for unsupervised video anomaly detection},
  author={Rakhmonov, Akhrorjon Akhmadjon Ugli and Subramanian, Barathi and Amirian Varnousefaderani, Bahar and Kim, Jeonghong},
  journal={ETRI Journal},
  volume={46},
  number={5},
  pages={890--903},
  year={2024},
  publisher={Wiley Online Library}
}
```

## **📬 Contact**  
For any questions or contributions, feel free to open an issue or reach out.  

---
