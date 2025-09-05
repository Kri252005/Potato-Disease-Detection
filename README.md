# 🥔 Potato Disease Detection (Early Blight, Healthy, Other)

This project uses **VGG16 (Transfer Learning)** with TensorFlow/Keras to detect **Early Blight disease in potato leaves**.  
It classifies potato leaf images into 3 categories:
- 🌱 Early_Blight
- 🌿 Healthy
- 🍂 Other (background or irrelevant leaves)

---

## 📂 Project Structure
Potato-Disease-Detection/
│── dataset/ # train/val split data
│ ├── train/
│ └── val/
│── dataset_original/ # original unprocessed dataset
│── model1.h5 # trained model
│── requirements.txt # dependencies
│── split_dataset.py # script to split dataset
│── train.py # training script
│── test.py # test on validation/test dataset
│── test_single.py # predict on a single image
│── app.py # Streamlit web app
│── README.md # project documentation

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Potato-Disease-Detection.git
   cd Potato-Disease-Detection
   
2.Create a virtual environment:
python -m venv venv
source venv/Scripts/activate   # On Windows
source venv/bin/activate       # On Linux/Mac

3.Install requirements:
pip install -r requirements.txt

4.🏋️ Training the Model
python train.py

5.🧪 Testing the Model
Evaluate on validation/test dataset
python test.py

6.Test a single image
python test_single.py

7.🌐 Running the Streamlit App
pip install streamlit
python -m streamlit run test.py


Acknowledgments
Dataset: PlantVillage Dataset
Pretrained Model: VGG16
Framework: TensorFlow/Keras


👩‍💻 Author

Krithi Mallika
📌 Computer Science Engineer | 🌱 AI & ML Enthusiast
