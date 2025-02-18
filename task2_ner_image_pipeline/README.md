### **📌 Task 2: Named Entity Recognition + Image Classification Pipeline**
This task builds a **multi-modal machine learning pipeline** that combines **Named Entity Recognition (NER) and Image Classification** to verify if a text description correctly matches an image.

### **🚀 Features**
✅ Implements two models:
- **Named Entity Recognition (NER)** (using `transformers`) to extract animal names from text.
- **Image Classification** (using `ResNet-18` in `torchvision.models`) to classify animals in images.

✅ Supports **end-to-end verification pipeline** that:
1. Extracts the **animal name** from text using **NER**.
2. Predicts the **animal in the image** using **CNN**.
3. Compares the results and outputs **True/False**.

✅ Includes a **Jupyter Notebook (`demo.ipynb`)** for step-by-step demonstration and edge case handling.

---

## **📁 Task Structure**
```
task2_ner_image_pipeline/
│── models/
│   │── image_classification/                            # Folder for NER model files
│   |    │── train_classifier.py             # Train Image Classification model
│   │── ner/                            # Folder for NER model files
|   │   │── train_ner.py                   # Train Named Entity Recognition model

│── pipeline.py             # Main verification pipeline script
│── dataset_prep.py             # Dataset Animal-10 preparation code
│── demo.ipynb                           # Jupyter Notebook for demonstration
│── README.md                            # Project documentation
│── requirements.txt                     # Dependencies
```

---

## **🔧 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Coin-Gulter/AI_DS_test.git
cd task2_ner_image_pipeline
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### ** Make kaggle directory **
```bash
mkdir kaggle
```

Get your api kaggle.json file from your kaggle account and change path to the file in the .env if needed.

---

## **📊 Dataset**
- **NER Dataset**: Trained on a subset of **CoNLL-2003** dataset with labels for animal names.
- **Image Dataset**: Uses **Animals-10 (Kaggle)** with 10 animal classes.

---

## **🛠 Dataset animal-10 preparation**
run:
```bash
python dataset_prep.py
```


## **🛠 Training the Models**
To train the **NER model**, run:
```bash
python models/ner/train_ner.py
```

To train the **Image Classification model**, run:
```bash
python models/image_classification/train_classifier.py
```

---

## **✅ Running the Verification Pipeline**
To test the pipeline, run:
```bash
python pipeline.py --text "There is a cat in the picture." --image "D:\Projects\Winstars_AI_DS_internship_test\task2_ner_image_pipeline\dataset\val\cow\OIP-_01xh7qI0nvypZLkOxGyIgHaFi.jpeg"
```

### **Pipeline Logic (`verification_pipeline.py`)**
1. **Extracts animal name** from text using NER.
2. **Predicts animal class** in the image using CNN.
3. **Compares the results** → Returns ✅ `True` (match) or ❌ `False` (mismatch).

---

## **📝 Code Overview**
### **1️⃣ Named Entity Recognition (NER)**
```python
inputs = ner_tokenizer(text, return_tensors="pt")
outputs = ner_model(**inputs).logits
predictions = torch.argmax(outputs, dim=-1)[0].tolist()
tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
```
- **Extracts words labeled as `B-Animal`** from the text.

---

### **2️⃣ Image Classification Model (CNN)**
```python
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)
with torch.no_grad():
    output = image_model(image)
predicted_class = torch.argmax(output, dim=1).item()
```
- **Predicts the most likely animal class** from the image.

---

## **📌 Edge Cases Considered**
1. **Multiple animals mentioned in text.**
2. **Misspelled or slightly different animal names.**
3. **Images with multiple animals.**
4. **No animal mentioned in text.**

