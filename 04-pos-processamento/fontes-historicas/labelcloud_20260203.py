import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import csv
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# # === CONFIGURAÇÃO DO PERFIL LOCAL DO CHROME ===
# profile_path = str(Path.home()) + "/AppData/Local/Google/Chrome/User Data"  # ajuste se necessário

# options = uc.ChromeOptions()
# options.add_argument(f"--user-data-dir={profile_path}")
# options.add_argument("--profile-directory=Profile 2")  # ou "Profile 1", etc
# options.add_argument("--start-maximized")
# # NÃO USAR headless! Deve estar visível para manter a sessão real

# driver = uc.Chrome(options=options, version_main=144)

options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")


driver = webdriver.Chrome(
    options=options
)

# === FUNÇÃO PARA EXTRAIR OS DADOS DA TABELA ===
def scrape_etherscan_label(url):
    driver.get(url)
    time.sleep(10)  # pode substituir por WebDriverWait

    data = []
    rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        if len(cols) >= 2:
            try:
                link = cols[0].find_element(By.TAG_NAME, 'a')
                href = link.get_attribute('href')
                address = href.split('/address/')[-1].strip()

                nametag = cols[1].text.strip()
                if address.startswith("0x"):
                    data.append((address, nametag))
            except Exception as e:
                print(f"⚠️ Erro ao processar linha: {e}")
    return data

# === URLs a serem raspadas ===
urls = {
    'selecao-inicial': 'https://etherscan.io/accounts/label/mev-builder?subcatid=undefined&size=200&start=0&col=1&order=asc',
    'mev-builder': 'https://etherscan.io/accounts/label/mev-builder?subcatid=undefined&size=200&start=0&col=1&order=asc',
    'mev-bot': 'https://etherscan.io/accounts/label/mev-bot?subcatid=undefined&size=5000&start=0&col=1&order=asc'
}

all_data = []

for label, url in urls.items():
    print(f"🔍 Coletando dados de {label}...")
    data = scrape_etherscan_label(url)
    print(f"✅ {len(data)} registros coletados de {label}.")
    all_data.extend(data)

driver.quit()

# === Salvar em CSV ===
with open('mev_addresses_with_tags.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['address', 'nametag'])
    writer.writerows(all_data)

print("📁 Arquivo salvo como mev_addresses_with_tags.csv")
