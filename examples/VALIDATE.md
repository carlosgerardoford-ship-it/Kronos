# validate.py: validación walk-forward de Kronos-small (CPU)
Compara la predicción con el baseline "el precio se queda igual" sobre ventanas consecutivas sin solape.
    python validate.py KO TLT BTCUSDT --interval 1h --lookback 400 --pred-len 24 --samples 1 --windows 20
    python validate.py KO --interval 1h --windows 40 --offset 480   # tramo anterior (confirmación)
    python validate.py KO --null-test                                # desfase con pares sin relación
Datos: Binance (cripto, pagina de a 1000 velas) y Yahoo Finance (acciones/ETF; sin 4h). Pesos: Hugging Face.
Salida en --out (def. outputs/validation): <TICKER>_<tramo>.csv por ticker (una fila por ventana, con trayectorias)
y summary_<tramo>.txt. Semilla por ventana = --seed + índice de ventana (def. 0).
Tests sin modelo: python test_validate.py
Desfase: el de best_lag (summary/null-test) es un artefacto (pares sin relación dan lo mismo). Usar: python analyze_lag.py outputs/validation 24
