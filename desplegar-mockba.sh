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

imprimir_error() {
    echo -e "${ROJO}❌ $1${NC}"
}

# Verificar si se ejecuta como root
if [ "$EUID" -eq 0 ]; then
    imprimir_error "Por favor no ejecutes como root. Usa un usuario normal."
    exit 1
fi

# Crear directorio del proyecto
DIRECTORIO_PROYECTO="$HOME/mockba-trader"
imprimir_estado "Creando directorio del proyecto: $DIRECTORIO_PROYECTO"
mkdir -p "$DIRECTORIO_PROYECTO"
cd "$DIRECTORIO_PROYECTO"

# Paso 1: Instalar Docker si no existe
if ! command -v docker &> /dev/null; then
    imprimir_advertencia "Docker no encontrado. Instalando..."
    curl -fsSL https://get.docker.com -o instalar-docker.sh
    sudo sh instalar-docker.sh
    sudo usermod -aG docker $USER
    imprimir_estado "Docker instalado correctamente"
else
    imprimir_estado "Docker ya está instalado"
fi

# Paso 2: Instalar Docker Compose si no existe
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    imprimir_advertencia "Docker Compose no encontrado. Instalando..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    imprimir_estado "Docker Compose instalado"
else
    imprimir_estado "Docker Compose ya está instalado"
fi

# Paso 3: Solicitar configuración al usuario
echo ""
echo "🔧 Configuración del Bot"
echo "========================"

# Solicitar API Keys
read -p "🔑 Ingresa tu BINANCE_API_KEY: " BINANCE_API_KEY
read -p "🔑 Ingresa tu BINANCE_SECRET_KEY: " BINANCE_SECRET_KEY
read -p "🤖 Ingresa tu DEEP_SEEK_API_KEY: " DEEP_SEEK_API_KEY

# Telegram 
read -p "🤖 Ingresa tu API_TOKEN de Telegram: " API_TOKEN
read -p "💬 Ingresa tu TELEGRAM_CHAT_ID: " TELEGRAM_CHAT_ID

# Idioma del bot
read -p "🌐 Idioma del bot (es/en) [es]: " BOT_LANGUAGE
BOT_LANGUAGE=${BOT_LANGUAGE:-es}

# Paso 4: Crear archivos de configuración
imprimir_estado "Creando archivos de configuración..."

# Crear docker-compose.yml
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
      - ./plantilla_prompt_llm.txt:/app/futures_perps/trade/binance/llm_prompt_template.txt

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

# Crear archivo .env con los valores proporcionados
cat > .env << EOF
# =============================================
# CONFIGURACIÓN DEL BOT MOCKBA TRADER
# =============================================

# CLAVES API DE BINANCE
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_SECRET_KEY=$BINANCE_SECRET_KEY

# CLAVE API DE DEEPSEEK
DEEP_SEEK_API_KEY=$DEEP_SEEK_API_KEY

# CONFIGURACIÓN DE TELEGRAM
API_TOKEN=$API_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

# CONFIGURACIÓN DEL BOT
BOT_LANGUAGE=$BOT_LANGUAGE
APP_PORT=8000

# CONFIGURACIÓN DE REDIS Y RENDIMIENTO
REDIS_URL=redis://localhost:6379/0
CPU_COUNT=0
MAX_WORKERS=10

# PARÁMETROS DE TRADING
RISK_PER_TRADE_PCT=1.5
MAX_LEVERAGE_HIGH=5
MAX_LEVERAGE_MEDIUM=4
MAX_LEVERAGE_SMALL=3
MICRO_BACKTEST_MIN_EXPECTANCY=0.0025
EOF

# Crear plantilla de prompt LLM
cat > plantilla_prompt_llm.txt << 'EOF'
Eres un trader experimentado de criptomonedas analizando datos de mercado para Binance Futures.

Por favor analiza estos datos y proporciona:
1. Sentimiento a corto plazo (Alcista/Bajista/Neutral)
2. Niveles clave de soporte y resistencia
3. Acción recomendada con nivel de confianza
4. Razonamiento breve

Mantén el análisis conciso y enfocado en insights accionables.
EOF

imprimir_estado "Archivos de configuración creados"

# Paso 5: Iniciar el bot
imprimir_estado "Iniciando Bot Mockba Trader..."
docker-compose up -d

echo ""
imprimir_estado "¡Bot iniciado correctamente!"
echo ""
echo "📊 Para ver logs: cd $DIRECTORIO_PROYECTO && docker-compose logs -f"
echo "🔧 Editar configuración: nano $DIRECTORIO_PROYECTO/.env"
echo "🛑 Detener bot: cd $DIRECTORIO_PROYECTO && docker-compose down"
echo "▶️  Iniciar bot: cd $DIRECTORIO_PROYECTO && docker-compose up -d"
echo ""
echo "💡 Configuración guardada en: $DIRECTORIO_PROYECTO/.env"
echo ""
imprimir_estado "¡Despliegue completado! 🎉"