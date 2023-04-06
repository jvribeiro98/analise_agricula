# -*- coding: utf-8 -*-
import ee
import datetime
import cv2
import numpy as np

# Inicialize a API do Earth Engine
ee.Initialize()

# Função para analisar a variação de cores
def analyze_color_variation(image, threshold=30):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean, stddev = cv2.meanStdDev(hsv_image)
    return stddev[0]

# Função para obter imagens de satélite usando a API do Earth Engine
def get_satellite_image(date, region):
    # Escolha a coleção de imagens e o período de tempo
    image_collection = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA") \
        .filterDate(date.strftime("%Y-%m-%d"), (date + datetime.timedelta(days=30)).strftime("%Y-%m-%d")) \
        .filterBounds(region)

    # Selecione a imagem mais recente
    image = ee.Image(image_collection.sort('system:time_start', False).first())

    # Aplique uma transformação para exibir a imagem corretamente
    image_rgb = image.select(['B4', 'B3', 'B2']).multiply(255)

    # Obtenha a URL da imagem
    image_url = image_rgb.getThumbURL({'region': region, 'dimensions': 512})

    return image_url

# Função para baixar imagens usando a URL
def download_image(url):
    response = requests.get(url)
    image_data = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image

# Parâmetros
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2022, 12, 31)
region = ee.Geometry.Rectangle([3.059,881821, -15,231027, -46,801146,1.003,7961598])  # Defina a região de interesse (longitude_min, latitude_min, longitude_max, latitude_max)

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
    print(f"Variação de cores para {current_date:%Y-%m}: {color_variation}")

    # Avance para o próximo mês
    current_date += datetime.timedelta(days=30)

