# AI-based Pneumonia detection tool and Generation of synthetic X-Ray images for OpenMRS 
It is an AI powered clinical decision support tool used to predict pneumpnia and its types which is integrated to OpenMRS EHR system. Synthetic X-Rays have been generated using DCGAN for demonstration purposes.

#Datasets
1. RSNA Pneumonia X-rays for Pneumonia detection tool : https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 
2. MIMIC-CXR-JPG from PhysioNet for Synthrtic X-Ray Generation : https://physionet.org/content/mimic-cxr-jpg/2.1.0/

# Requirements
1. numpy
2. pandas
3. Flask
4. scikit-learn
5. https://download.pytorch.org/whl/cpu/torch-1.7.0%2Bcpu-cp36-cp36m-linux_x86_64.whl
6. https://download.pytorch.org/whl/cpu/torchvision-0.8.1%2Bcpu-cp36-cp36m-linux_x86_64.whl
7. requests
8. gunicorn == 20.0.4
9. Flask==2.0.2
10. pandas==1.3.3
11. numpy
12. requests==2.31.0
13. Pillow==9.5.0
14. tensorflow==2.5.0
15. Werkzeug==2.2.3
16. opencv-python==4.6.0.66
17 seaborn==0.11.2
18. matplotlib==3.4.3
19. scikit-learn
20. reportlab==3.6.12
21. PyMySQL==0.10.0
22. qrcode==7.4.2


# Usage
Steps
1. Install OpenMRS standalone version.
2. Clone the repositry and run python app.py from the environment created to run the pneumonia detection tool.

# Related work used for generation of X-Ray images 
https://www.kaggle.com/code/djibybalde/dcgan-keras-chest-x-ray-images

