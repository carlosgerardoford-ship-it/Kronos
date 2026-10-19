# Criterios fijados antes de correr (2026-10-19)
Configuración: Kronos-small, 1h, lookback 400, pred-len 24. Activos: BTCUSDT (200 ventanas) y QQQ (126, máximo disponible en Yahoo). Un segundo juego con --samples 5 y 60 ventanas.
Dirección: pasa si (a) le gana al baseline "se queda igual" en más de la mitad de las ventanas y (b) su % de aciertos supera la base de apostar siempre al lado más frecuente, con el límite inferior del IC95% (Wilson) por encima de 50%.
Si pasa, confirmación en un tramo anterior (--offset) con 100 ventanas. Si no pasa, no se prueban más activos ni temporalidades.
Volatilidad: el modelo es útil si su correlación (Spearman) con el rango real de la ventana supera a la del baseline "misma volatilidad que en el contexto" y su error log mediano es menor.
