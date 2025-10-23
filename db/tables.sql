-- Table: scalp_positions
CREATE TABLE IF NOT EXISTS public.scalp_positions
(
    id integer NOT NULL DEFAULT nextval('scalp_positions_id_seq'::regclass),
    chat_id bigint NOT NULL,
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    side character varying(10) COLLATE pg_catalog."default" NOT NULL,
    entry_price numeric(20,8) NOT NULL,
    stop_loss numeric(20,8) NOT NULL,
    take_profit numeric(20,8) NOT NULL,
    quantity numeric(20,8) NOT NULL,
    notional_usd numeric(20,2),
    status character varying(20) COLLATE pg_catalog."default" DEFAULT 'OPEN'::character varying,
    current_pnl_pct numeric(10,4) DEFAULT 0.0,
    current_pnl_usd numeric(20,2) DEFAULT 0.0,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    entry_order_id bigint,
    tp_order_id bigint,
    sl_order_id bigint,
    fill_price numeric(20,8),
    exchange text COLLATE pg_catalog."default",
    CONSTRAINT scalp_positions_pkey PRIMARY KEY (id)
)


-- Table: scalp_settings
CREATE TABLE IF NOT EXISTS scalp_settings (
    chat_id BIGINT PRIMARY KEY,
    signals_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);