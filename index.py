# -*- coding: utf-8 -*-
import ee
import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import requests

# Inicialize a API do Earth Engine
ee.Initialize()

# Definindo a arquitetura da rede neural
class PlantColorationNet(nn.Module):
    def __init__(self):
        super(PlantColorationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Função para processar a imagem de entrada e fazer a previsão
def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantColorationNet()
    model.load_state_dict(torch.load("modelo.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    labels = ["Normal", "Mudança de coloração"]
    return labels[predicted.item()]

# Função para analisar a variação de cores
def analyze_color_variation(image):
    result = predict(image)
    return result

# Função para obter imagens de satélite usando a API do Earth Engine
def get_satellite_image(date, region):
    # Escolha a coleção de imagens e o período de tempo
    image_collection = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA") \
        .filterDate(date.strftime("%Y-%m-%d"), (date + datetime.timedelta(days=30)).strftime("%Y-%m-%d")) \
        .filterBounds(region)

    # Verifique se a coleção tem alguma imagem
    if image_collection.size().getInfo() == 0:
        print(f"Nenhuma imagem disponível para {date:%Y-%m}. Usando uma imagem padrão.")
        image = ee.Image("LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318")
    else:
        # Selecione a imagem mais recente
        image = ee.Image(image_collection.sort('system:time_start', False).first())

    # Aplique uma transformação para exibir a imagem corretamente
    image_rgb = image.select(['B4', 'B3', 'B2']).multiply(255)

    # Obtenha a URL da imagem
    image_url = image_rgb.getThumbURL({
        'region': region,
        'dimensions': 512,
        'min': 0,
        'max': 255,
        'format': 'png'
    })

    return image_url



    # Selecione a imagem mais recente
    image = ee.Image(image_collection.sort('system:time_start', False).first())

    # Aplique uma transformação para exibir a imagem corretamente
    image_rgb = image.select('B4', 'B3', 'B2').multiply(255)

    # Obtenha a URL da imagem
    image_url = image_rgb.getThumbURL({'region': region, 'dimensions': 512})

    return image_url

# Função para baixar imagens usando a URL
def download_image(url):
    response = requests.get(url)
    image_data = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Parâmetros
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2022, 12, 31)
region = ee.Geometry.Rectangle([-46.801146, -15.231027, 3.059, 1.003])  # Defina a região de interesse (longitude_min, latitude_min, longitude_max, latitude_max)

# Loop mensal
current_date = start_date
while current_date <= end_date:
    print(f"Analisando imagens de {current_date:%Y-%m}")

    # Obtenha a imagem de satélite
    image_url = get_satellite_image(current_date, region)
    satellite_image = download_image(image_url)

    # Analise a variação de cores na imagem
    color_variation = analyze_color_variation(satellite_image)

    # Exiba o resultado
    print(f"Resultado para {current_date:%Y-%m}: {color_variation}")

    # Avance para o próximo mês
    current_date += datetime.timedelta(days=30)

