**Viral Meme Classifier: Understanding Sentiment and Content in Social Media Memes**

---

## **Project Overview**

This project focuses on analyzing meme sentiments (Positive or Negative) using Machine Learning techniques. The system processes meme images by extracting text, analyzing sentiment, and classifying the sentiment using a trained Random Forest model.

---

## **Files Included**

1. **Notebook**:  
   * [Google Colab Notebook](https://colab.research.google.com/drive/1SFM9ictFjVSvBFWxl6_XhrQA3BJJtqg0#scrollTo=Kt81QfJaV1Iz)  
   * `Final_Project.ipynb` (downloadable version of the notebook)  
2. **Trained Models**:  
   * `RF_model.pkl`: Pre-trained Random Forest model.  
   * `vectorizer.pkl`: TF-IDF vectorizer for text transformation.  
3. **Dataset**:  
   * `memegenerator.csv`: Contains meme text and metadata for training and evaluation.  
4. **Sample Images**:  
   * `positive.jpeg`: Example of a Positive sentiment meme.  
   * `negative.jpeg`: Example of a Negative sentiment meme.  
5. **Outputs**:  
   * `model_predictions.csv`: Predictions for testing data.  
6. **Documentation**:  
   * Project Report (PDF): Detailed explanation of the project.  
   * Presentation Slides: Key highlights for presenting the project.

---

## 

## **Requirements**

Ensure the following dependencies are installed:

1. Python 3.8 or later

Required Python libraries:

`pip install pytesseract Pillow torch torchvision scikit-learn nltk joblib`

2. **Tesseract OCR**:  
   * Install Tesseract on your system: [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract).

---

## **Setup Instructions**

### **1\. Clone or Download the Repository**

* Clone the repository or download all files to your local system.

### **2\. Run the Notebook**

* Open the `Final_Project.ipynb` notebook in [Google Colab](https://colab.research.google.com/) or your local Jupyter environment.

### **3\. Upload Required Files**

* Upload the following files to your notebook runtime:  
  * `RF_model.pkl`  
  * `vectorizer.pkl`  
  * `X_train_vec.pkl`  
  * `memegenerator.csv`  
  * `meme_predictions.csv`  
  * `positive.jpeg`, `negative.jpeg`

### **4\. Test the System**

Replace the image path in the notebook code with the meme you want to test:

`image_path = 'positive.jpeg'`

* Run the notebook and observe:  
  * Extracted text from the image.  
  * Detected sentiment (Positive or Negative).  
  * Visualization of the ROC Curve.

---

## 

## **How to Train Your Model**

1. Modify and execute the training section in the provided notebook to train your own model.  
2. Save the trained model and vectorizer using `joblib.dump()` for future use.

---

## **Contact**

For queries or issues, feel free to reach out.

