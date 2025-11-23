#!/bin/bash

echo "🤖 Bot Mockba Trader - Despliegue Automático"
echo "============================================"

# Colores para output
ROJO='\033[0;31m'
VERDE='\033[0;32m'
AMARILLO='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir con color
imprimir_estado() {
    echo -e "${VERDE}✅ $1${NC}"
}

imprimir_advertencia() {
    echo -e "${AMARILLO}⚠️  $1${NC}"
}

# Crear directorio del proyecto
DIRECTORIO_PROYECTO="/opt/mockba-trader"
imprimir_estado "Creando directorio del proyecto: $DIRECTORIO_PROYECTO"
mkdir -p "$DIRECTORIO_PROYECTO"
cd "$DIRECTORIO_PROYECTO"

# Paso 1: Instalar Docker si no existe
if ! command -v docker &> /dev/null; then
    imprimir_advertencia "Docker no encontrado. Instalando..."
    curl -fsSL https://get.docker.com -o instalar-docker.sh
    sh instalar-docker.sh
    imprimir_estado "Docker instalado correctamente"
else
    imprimir_estado "Docker ya está instalado"
fi

# Paso 2: Instalar Docker Compose si no existe
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    imprimir_advertencia "Docker Compose no encontrado. Instalando..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    imprimir_estado "Docker Compose instalado"
else
    imprimir_estado "Docker Compose ya está instalado"
fi

# Paso 3: Descargar archivos de configuración
imprimir_estado "Descargando archivos de configuración..."

# Descargar docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  micro-mockba-binance-futures-bot:
    image: andresdom2004/micro-mockba-binance-futures-bot:latest
    container_name: micro-mockba-binance-futures-bot
    restart: always
    env_file: 
      - .env
    volumes:
      - ./.env:/app/.env
      - ./prompt.txt:/app/futures_perps/trade/binance/llm_prompt_template.txt

  watchtower:
    image: containrrr/watchtower
    container_name: watchtower-binance
    restart: always
    depends_on:
      - micro-mockba-binance-futures-bot
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - WATCHTOWER_CLEANUP=true
      - WATCHTOWER_POLL_INTERVAL=300
      - WATCHTOWER_LIFECYCLE_HOOKS=true
      - WATCHTOWER_LABEL_ENABLE=true
EOF

# Crear archivo .env
cat > .env << 'EOF'
BINANCE_API_KEY=tu_api_key_de_binance
BINANCE_SECRET_KEY=tu_secret_key_de_binance
DEEP_SEEK_API_KEY=tu_clave_de_deepseek
API_TOKEN=tu_token_de_telegram
TELEGRAM_CHAT_ID=tu_chat_id_de_telegram
BOT_LANGUAGE=es
APP_PORT=8000
RISK_PER_TRADE_PCT=1.5
EOF

# Crear prompt en español
cat > prompt.txt << 'EOF'
Analiza este dataset de trading. Basado en estos datos, ¿debería tomar la señal sugerida? ¿Ves patrones técnicos que confirmen? ¿Niveles clave de soporte/resistencia? ¿El order book muestra liquidez suficiente?
EOF

imprimir_estado "Archivos de configuración creados"

# Paso 4: Iniciar el bot automáticamente
imprimir_estado "Iniciando Bot Mockba Trader..."
docker-compose up -d

echo ""
imprimir_estado "¡Bot iniciado correctamente!"
echo ""
echo "📊 Para ver logs: docker-compose logs -f"
echo "🌐 Panel de control: http://localhost:8000"
echo "🔧 Editar configuración: nano $DIRECTORIO_PROYECTO/.env"
echo "🛑 Detener bot: docker-compose down"
echo "▶️  Iniciar bot: docker-compose up -d"
echo ""
echo "💡 Recuerda editar el archivo .env con tus claves API:"
echo "   nano $DIRECTORIO_PROYECTO/.env"
echo ""
imprimir_estado "¡Despliegue completado! 🎉"