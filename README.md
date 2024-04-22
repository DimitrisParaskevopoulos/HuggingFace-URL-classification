# HuggingFace-text-classification

### Description

**nlp_classification.ipynb**: Exploring ML models accuracy for text classification through systematic experiments with varying features and models (8 ML modes, HuggingFace Transformers). Achieved up to 95% accuracy, revealing key insights on optimal input-model combos

**Deployment**: Deploy the Random Forest utilising FastAPI and Docker
### Getting Started

Step-by-step instructions for setting up and running the app

#### Setting Up
 ```
git clone https://github.com/DimitrisParaskevopoulos/HuggingFace-text-classification.git
 ```
 ```
cd your-local-path/HuggingFace-text-classification
 ```

#### Running
 ```
docker-compose up --build -d 
 ```

#### Testing
 ```
curl --location "http://localhost:5000/predict/rf" ^
--header "Content-Type: application/json" ^
--data "{""url"": ""https://www.sport24.gr/football/real-mpartselona-3-2-o-vasilias-mpeligcham-milise-sto-91-kai-estepse-protathlites-toys-merengkes.10302657.html""}"
 ```
