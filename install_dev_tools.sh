#!/bin/bash

# set -e

# Лог файл з версіями встановлених пакетів
log_file='install.log'

# Визначаємо мінімальну потрібну версію Python
required_version="3.9"

# Директорія для віртуального середовища
VENV_DIR=".venv"


######## docker installation test ########
# Перевірка, чи встановлена команда docker
if ! command -v docker &> /dev/null
then
    echo "Docker не знайдений, виконується встановлення..."
    
    # Uninstall old versions
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    # Install Docker
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker

    echo "Docker успішно встановлено"
else
    echo "Docker вже встановлено"
fi

# Check Docker version
sudo docker --version >> $log_file
docker compose version >> $log_file



######## python3 v. 3.9 or later installation test ########

# Функція порівняння версій
function version_get() {
  # Порівнює дві версії у форматі X.Y
  printf '%s\n%s\n' "$1" "$2" | sort -C -V
}

# Отримуємо встановлену версію python3
if command -v python3 &>/dev/null; then
  installed_version=$(python3 --version 2>&1 | awk '{print $2}')
else
  installed_version=""
fi

# Перевірка чи версія >= 3.9
if [[ -z "$installed_version" ]] || version_get "$installed_version" "$required_version"; then
  echo "Python версії 3.9 або вище не встановлено або версія менша за 3.9."
  echo "Встановлення Python..."

  # Приклад для Debian/Ubuntu, можна додати інші дистрибутиви зі своїми командами
  sudo apt update
  sudo apt install -y python3
else
  echo "Python версії $installed_version вже встановлено."
fi

python3 --version >> $log_file



######## pip installation test ########

# Перевірка наявності pip
if ! command -v pip > /dev/null 2>&1; then
  echo "pip не знайдено, встановлюємо..."
  sudo apt update
  sudo apt install -y python3-pip
else
  echo "pip вже встановлений"
fi 

pip --version  >> $log_file



######### venv creation test ########


# Перевірка, чи існує директорія віртуального середовища
if [ -d "$VENV_DIR" ]; then
  echo "Віртуальне середовище вже існує."
else
  echo "Віртуальне середовище не знайдено. Створюємо..."
  python3 -m venv "$VENV_DIR"
  echo "Віртуальне середовище створене."
fi

source "$VENV_DIR/bin/activate"
echo "Віртуальне середовище активоване."



######## torch, torchvision, pillow installation test ########

# Масив пакетів, які потрібно перевірити та встановити
packages=("torch" "torchvision" "Pillow" "Django")

for pkg in "${packages[@]}"; do

    # деякі пакети чутливі до регістру
    ((pip list | grep "${pkg,,}") || (pip list | grep $pkg)) > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "Пакет $pkg вже встановлений."
    else
        echo "Пакет $pkg не встановлений. Встановлення..."
        pip install -U "${pkg,,}"
    fi

    ((pip list | grep "${pkg,,}") || (pip list | grep $pkg)) >> $log_file
done

deactivate
echo "Віртуальне середовище деактивоване."
