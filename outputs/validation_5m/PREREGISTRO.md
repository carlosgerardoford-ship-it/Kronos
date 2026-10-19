# Criterios fijados antes de correr (5m)
Configuración: Kronos-small, 5m, lookback 400, pred-len 120 (la del ejemplo oficial del repo), 1 muestra. Activos: BTCUSDT (100 ventanas) y QQQ (35, máximo disponible en Yahoo para 5m).
Mismos criterios que en outputs/validation_long/PREREGISTRO.md:
Dirección: pasa si (a) le gana al baseline "se queda igual" en más de la mitad de las ventanas y (b) su % de aciertos supera la base de apostar siempre al lado más frecuente, con el límite inferior del IC95% (Wilson) por encima de 50%.
Volatilidad: útil si su correlación (Spearman) con el rango real supera a la del baseline "misma volatilidad que en el contexto" y su error log mediano es menor.
Si no pasa, no se prueban más activos ni temporalidades. QQQ con 35 ventanas tiene poca potencia: solo un resultado muy alto podría pasar el criterio.
