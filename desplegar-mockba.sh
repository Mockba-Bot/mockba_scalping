#!/bin/bash

echo "🤖 Bot Mockba Trader - Despliegue Automático"
echo "============================================"

# Colores
ROJO='\033[0;31m'
VERDE='\033[0;32m'
AMARILLO='\033[1;33m'
AZUL='\033[0;34m'
NC='\033[0m' # No Color

imprimir_estado() { echo -e "${VERDE}✅ $1${NC}"; }
imprimir_advertencia() { echo -e "${AMARILLO}⚠️  $1${NC}"; }
imprimir_error() { echo -e "${ROJO}❌ $1${NC}"; }
imprimir_info() { echo -e "${AZUL}💡 $1${NC}"; }

# Helper: read required input (must be non-empty)
read_required() {
    local prompt="$1"
    local var_name="$2"
    while true; do
        read -p "$prompt ('c' para cancelar): " input
        case "$input" in
            c|C)
                imprimir_info "Instalación cancelada por el usuario."
                exit 0
                ;;
            "")
                imprimir_advertencia "Este campo es obligatorio. Por favor, ingrésalo."
                ;;
            *)
                eval "$var_name='$input'"
                return
                ;;
        esac
    done
}

# Helper: read optional input (Enter/x = default, c = cancel)
read_optional() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    read -p "$prompt (Enter o 'x' para usar '$default', 'c' para cancelar): " input
    case "$input" in
        c|C)
            imprimir_info "Instalación cancelada por el usuario."
            exit 0
            ;;
        ""|"x"|"X")
            eval "$var_name='$default'"
            ;;
        *)
            eval "$var_name='$input'"
            ;;
    esac
}

# === Main Script ===

DIRECTORIO_PROYECTO="/opt/mockba-trader"
imprimir_estado "Creando directorio del proyecto: $DIRECTORIO_PROYECTO"
mkdir -p "$DIRECTORIO_PROYECTO"
cd "$DIRECTORIO_PROYECTO" || { imprimir_error "No se pudo acceder a $DIRECTORIO_PROYECTO"; exit 1; }

# === Docker ===
if ! command -v docker &> /dev/null; then
    imprimir_advertencia "Docker no encontrado. Instalando..."
    curl -fsSL https://get.docker.com -o instalar-docker.sh
    sh instalar-docker.sh
    imprimir_estado "Docker instalado correctamente"
else
    imprimir_estado "Docker ya está instalado"
fi

# === Docker Compose ===
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    imprimir_advertencia "Docker Compose no encontrado. Instalando..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    imprimir_estado "Docker Compose instalado"
else
    imprimir_estado "Docker Compose ya está instalado"
fi

# === Configuración ===
imprimir_info "Configuración del bot: API Keys (obligatorias)"
read_required "🔑 BINANCE_API_KEY" BINANCE_API_KEY
read_required "🔑 BINANCE_SECRET_KEY" BINANCE_SECRET_KEY
read_required "🤖 DEEP_SEEK_API_KEY" DEEP_SEEK_API_KEY

echo
imprimir_info "Configuración opcional: Notificaciones por Telegram"
read_optional "🤖 Telegram API_TOKEN" "" API_TOKEN
read_optional "💬 TELEGRAM_CHAT_ID" "" TELEGRAM_CHAT_ID

echo
imprimir_info "Configuración del bot"
read_optional "🌐 Idioma (es/en)" "es" BOT_LANGUAGE
read_optional "📊 Riesgo por trade (%)" "1.5" RISK_PER_TRADE_PCT
read_optional "🎚️ Apalancamiento alto" "5" MAX_LEVERAGE_HIGH
read_optional "🎚️ Apalancamiento medio" "4" MAX_LEVERAGE_MEDIUM
read_optional "🎚️ Apalancamiento bajo" "3" MAX_LEVERAGE_SMALL
read_optional "📈 Expectativa mínima backtest" "0.0025" MICRO_BACKTEST_MIN_EXPECTANCY

echo
imprimir_info "Prompt de IA (deja en blanco para usar el predeterminado)"
DEFAULT_PROMPT="Analiza este dataset de trading. Basado en estos datos, ¿debería tomar la señal sugerida? ¿Ves patrones técnicos que confirmen? ¿Niveles clave de soporte/resistencia? ¿El order book muestra liquidez suficiente?"
read_optional "✏️ Tu prompt personalizado" "$DEFAULT_PROMPT" PROMPT_PERSONALIZADO

# === Guardar archivos ===
imprimir_estado "Creando archivos de configuración..."

# docker-compose.yml
cat > docker-compose.yml << EOF
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

# .env
cat > .env << EOF
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_SECRET_KEY=$BINANCE_SECRET_KEY
DEEP_SEEK_API_KEY=$DEEP_SEEK_API_KEY
API_TOKEN=$API_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
BOT_LANGUAGE=$BOT_LANGUAGE
APP_PORT=8000
RISK_PER_TRADE_PCT=$RISK_PER_TRADE_PCT
MAX_LEVERAGE_HIGH=$MAX_LEVERAGE_HIGH
MAX_LEVERAGE_MEDIUM=$MAX_LEVERAGE_MEDIUM
MAX_LEVERAGE_SMALL=$MAX_LEVERAGE_SMALL
MICRO_BACKTEST_MIN_EXPECTANCY=$MICRO_BACKTEST_MIN_EXPECTANCY
EOF

# prompt.txt
echo "$PROMPT_PERSONALIZADO" > prompt.txt

imprimir_estado "Archivos creados: .env, docker-compose.yml, prompt.txt"

# === Iniciar ===
imprimir_estado "Iniciando el bot con Docker Compose..."
if command -v docker-compose &> /dev/null; then
    DOCKER_CMD="docker-compose"
else
    DOCKER_CMD="docker compose"
fi

$DOCKER_CMD up -d

if [ $? -eq 0 ]; then
    echo
    imprimir_estado "✅ ¡Bot iniciado correctamente!"
    echo
    echo "📝 Editar configuración:   nano $DIRECTORIO_PROYECTO/.env"
    echo "✏️  Editar prompt:        nano $DIRECTORIO_PROYECTO/prompt.txt"
    echo "📊 Ver logs:              $DOCKER_CMD logs -f"
    echo "🛑 Detener bot:           $DOCKER_CMD down"
    echo "▶️  Iniciar bot:          $DOCKER_CMD up -d"
    echo
    imprimir_estado "¡Despliegue completado! 🎉"
else
    imprimir_error "❌ Falló al iniciar el contenedor. Revisa la configuración."
    exit 1
fi