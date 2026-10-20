# Prerregistro: Kronos-small como generador de distribuciones (BTCUSDT 1h)

Fecha: 2026-10-20. Escrito ANTES de correr el modelo en modo probabilístico. Los criterios no se cambian después de ver resultados.
Pregunta: ¿las trayectorias muestreadas por Kronos-small (sin ajuste fino) forman una distribución mejor y calibrada, comparada con baselines estadísticos?

## Configuración
- Activo y datos: BTCUSDT, 1h, Binance spot (data-api.binance.vision), solo velas cerradas. Se usan las últimas 400 + 100·24 = 2800 velas disponibles al momento de correr; el período exacto se registra en el informe.
- Ventanas: 100, walk-forward, lookback 400, pred_len 24. La parte a predecir no se solapa entre ventanas y cada ventana solo ve velas anteriores a ella (sin fuga de datos).
- Modelo: Kronos-small + Kronos-Tokenizer-base, sin ajuste fino. T=1.0, top_p=0.9, top_k=0.
- **Muestras de Kronos: 25 por ventana** (trayectorias individuales, sin promediar, obtenidas con un lote de 25 copias del contexto y `sample_count=1`).
- Semilla de Kronos = semilla_base + índice de ventana (semilla_base = 0). Semillas de baselines = semilla_base + índice de ventana + offset fijo por baseline.
- Todo se evalúa en log-retornos acumulados respecto del último cierre del contexto: r_h = ln(cierre_h / último cierre), h = 1..24. Caminos y eventos usan la serie de cierres.

### Nota sobre el cambio de 50 a 25 muestras
El plan original usaba 50 muestras por ventana. Se midió el tiempo con 3 ventanas piloto (BTCUSDT 1h, 50 muestras): ~372 s por ventana, ≈ 10,3 h para 100 ventanas. Se decidió bajar a **25 muestras (≈ 5,2 h)** por tiempo de cómputo, el 2026-10-20 y **antes de ver ningún resultado de desempeño**: de esas ventanas piloto solo se miró el tiempo y la dispersión entre muestras, no ninguna métrica de acierto. Las 100 ventanas se mantienen.

## Baselines probabilísticos (1000 trayectorias por ventana; usan SOLO el contexto)
Los retornos del contexto son los 399 log-retornos horarios de las 400 velas.
1. Paseo aleatorio normal: media 0, desvío = desvío muestral (ddof=1) de los retornos del contexto.
2. Paseo aleatorio t de Student: media 0; grados de libertad y escala por máxima verosimilitud sobre los retornos del contexto (scipy.stats.t.fit con loc=0); gl acotados a [2.1, 100].
3. Bootstrap por bloques: cada trayectoria es un bloque de 24 retornos consecutivos del contexto, con inicio uniforme entre las 376 posiciones posibles (bloque configurable). Con un bloque de 24 y pred_len 24, equivale a una **simulación histórica de trayectorias de 24 h tomadas del contexto**.
4. GARCH(1,1): paquete `arch`, media cero, innovaciones normales, ajustado sobre los retornos del contexto (en %), simulación hacia adelante de 24 pasos con 1000 simulaciones. Si el ajuste falla (o no converge), la ventana usa el paseo normal como respaldo, y **el resumen reporta cuántas ventanas de GARCH usaron ese respaldo**.

## Dependencias
Además de `requirements.txt`: `arch` y `scipy` (versiones usadas: arch 8.0.0, scipy 1.18.1), listadas en `examples/requirements_prob.txt` (`pip install -r examples/requirements_prob.txt`).

## Estimador de CRPS (igual para el modelo y los baselines)
CRPS "justo" con corrección de muestra finita: CRPS = (1/m)·Σ|x_i − y| − (1/(2·m·(m−1)))·Σ_i Σ_j |x_i − x_j|, con m = 25 para Kronos y m = 1000 para cada baseline. El mismo estimador se usa para el cierre final y para los horizontes 1, 6, 12 y 24. El energy score de la trayectoria usa el análogo justo con la norma euclídea.

## Criterios de aprobación (primarios)
Pasa solo si se cumplen (a) Y (b). Si no pasa, no se prueban otros activos ni temporalidades.
- **(a) Habilidad:** skill score del CRPS del cierre final > 0 contra el MEJOR baseline, y test de Diebold-Mariano sobre las diferencias de CRPS por ventana con p < 0,05 de una cola.
  - Skill score = 1 − (CRPS medio del modelo) / (CRPS medio del baseline). El "mejor baseline" es el de menor CRPS medio del cierre final sobre las 100 ventanas (se elige después, lo que lo hace más exigente, no menos).
  - Diferencial por ventana d = CRPS_baseline − CRPS_modelo (positivo = gana el modelo). H0: E[d] = 0 contra H1: E[d] > 0. Estadístico con varianza HAC de Newey-West, núcleo de Bartlett, ancho de banda 4, p de una cola con aproximación normal. Como verificación de robustez se informa también el p de un bootstrap por bloques (bloques de 4 ventanas, 10.000 remuestreos); el criterio decisivo es el de Newey-West.
- **(b) Calibración no rechazada** (sobre el cierre final de Kronos):
  - Cobertura de los intervalos centrales 50%, 80% y 90% (construidos con cuantiles empíricos de las 25 muestras): el test de Kupiec (razón de verosimilitud de cobertura incondicional) no rechaza en ninguno de los tres niveles (p > 0,05 en cada uno).
  - Histograma PIT: el valor PIT es aleatorizado, PIT = (nº de muestras < real + U) / (m + 1), con U uniforme (semillado), para evitar la discretización con 25 muestras. Kolmogorov-Smirnov contra U(0,1): no rechaza (p > 0,05).

## Métricas secundarias (se reportan; NO definen la aprobación)
- CRPS por horizonte (1, 6, 12, 24) y skill score contra cada baseline; ancho medio de los intervalos 50/80/90%; cobertura de los baselines.
- Energy score de la trayectoria completa.
- Eventos de camino (sobre la serie de cierres, retorno simple respecto del último cierre del contexto): toca +1%, −1%, +2%, −2%, +3%, −3% y drawdown > 2% (caída desde el máximo corrido, que parte del último cierre). La probabilidad predicha es la fracción de trayectorias con el evento. Brier score contra los baselines, **sin recortar probabilidades**: si la probabilidad predicha es exactamente 0 o 1 se usa tal cual. Se documenta, por evento, en cuántas ventanas la probabilidad predicha de Kronos quedó en 0 o en 1.
- Colas (cola izquierda del retorno final): VaR (cuantil) y ES (media por debajo del VaR) al 5% y al 1%; Kupiec y Christoffersen (independencia y cobertura condicional) sobre las excedencias.
  - **VaR/ES al 1% con 25 muestras no es estimable de forma confiable** (cae fuera del rango de las muestras): se reporta marcado como "no confiable" y queda excluido de cualquier conclusión. El VaR/ES al 5% se mantiene.
- Régimen: correlación de Spearman entre la dispersión predicha (desvío entre trayectorias del retorno final) y el |retorno final| realizado, modelo contra GARCH.

## Salidas
`outputs/validation_prob/`: `BTCUSDT_1h_prob.csv` (una fila por ventana, con checkpoint), `samples_BTCUSDT_1h.npz` (trayectorias de Kronos, con checkpoint incremental y posibilidad de reanudar), `summary_prob.txt` (tabla, tests y veredicto contra este prerregistro), `pit_hist.png`, `coverage.png`. El informe final: `INFORME_PROB_KRONOS.txt`.

## Adenda (2026-10-20, escrita antes de correr el modelo; detalle de implementación, no cambia los criterios)
Los tests sin modelo mostraron que el cuantil "lineal" por defecto de numpy, con 25 muestras, da intervalos que cubren ~83% cuando el nominal es 90% (~74% para 80%, ~46% para 50%) incluso con un pronóstico perfectamente calibrado, y Kupiec lo rechazaría. Por eso los cuantiles empíricos (intervalos 50/80/90% y VaR/ES) se calculan con el estimador de Weibull (posición q·(m+1)), que da cobertura exacta bajo intercambiabilidad; queda un residuo de ~+1 punto de sobrecobertura en el 90% por interpolar en las colas. Se usa el mismo estimador para Kronos y para los baselines. Los criterios (a) y (b) y sus umbrales no cambian.
Además, la simulación de GARCH usa el generador `rng` de `arch` (su parámetro `random_state` se ignora), para que sea reproducible con la misma semilla.
