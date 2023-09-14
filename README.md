# 🐞 nanovector: Efficient Vector Database with NumPy

nanovector is a lightweight, NumPy-powered vector database. It simplifies vector data storage and retrieval, making it perfect for embeddings and high-dimensional data.

<div align="center">
<img src="https://cdn.discordapp.com/attachments/933694512960774185/1135229083291226232/image.png" width="50%">
</div>

## 🙋‍♀️ Why nanovector?

**Vector databases**, like nanovector, are essential for:

1. **High-Dimensional Data:** Storing complex, multi-dimensional vectors efficiently.

2. **Distance Calculations:** Quickly find similarities or distances between vectors.

3. **Scalability:** Handle large datasets effortlessly.

nanovector excels because it's:

- **Lightweight:** Minimalist and easy to integrate, 
  
- **Efficient:** Speedy storage and retrieval using barebone numpy.

- **Dockerized:** Easily deploy Nanovector in containerized environments.

- **Direct Text to Vector Pipeline:** Seamlessly convert text data into vectors within the database.

- **Versatile:** Ideal for embeddings, features, and more.

- **Open Source:** Customizable and transparent.

## 🎨 Features
Nanovector offers the following key features:
- Efficient vector storage and retrieval.
- Dockerized for easy deployment.
- Direct pipeline for converting text to vectors.
- Integration with Sentence Transformers for powerful embeddings.
- PCAIndex for scaling high dimensional embeddings.

## 🍥 System Design

<img width="75%" alt="image" src="https://github.com/MananSuri27/nanovector/assets/84636031/10a1e1b8-816e-4041-98b8-18a85a5ff977">

## ⛏️ Set-up
### 🐳 Docker
Assuming you have docker installed, you can easily use docker to setup the vector server.

1. Pull the Docker image from Docker Hub:
   ```bash
   docker pull manansuri27/nanovector
   ```
   
2. Run the image now,
   ```bash
   docker run manansuri27/nanovector
   ```
The server will be running on `localhost:5000` now.

### 🐙 GitHub
Follow the steps below to setup the repository and run the server.

1. Clone the repo
   ```bash
   git clone https://github.com/MananSuri27/nanovector.git
   cd nanovector
   ```
3. Create a conda environment, and activate it
   ```bash
   conda create -n nanovector
   conda activate nanovector
   ```
5. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
7. Run the server
   ```bash
   python3 -m app.app
   ```
The server will be running on `localhost:5000` now.

## 📜 API Documentation
API Documentation is available in the [app directory](https://github.com/MananSuri27/nanovector/blob/main/app/README.md).
## 🧪 Testing
Once you have `pytest` installed, test using:
```bash
pytest
```

## 🍼 TODO
- [ ] Add tests for PCAIndex
- [ ] Include more similarity metrics


## 📇 Contact
Contact me on [Linkedin](https://www.linkedin.com/in/manansuri27/), drop an [email](mailto:manansuri27@gmail.com), or check me out on my website [manansuri.com](https://manansuri.com/).
