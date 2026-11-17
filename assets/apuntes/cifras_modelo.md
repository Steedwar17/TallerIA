# Cifras del modelo — extracción trazable

| Cifra | Valor exacto | Ubicación |
|---|---:|---|
| Split train IMDB | 25.000 reseñas | Notebook, celda 5 y 7 |
| Split test IMDB | 25.000 reseñas | Notebook, celda 5 y 7 |
| Split `unsupervised` no usado | 50.000 reseñas | Notebook, celda 5 |
| Train: negativas / positivas | 12.500 / 12.500 (50,0% / 50,0%) | Notebook, celda 8 |
| Test: negativas / positivas | 12.500 / 12.500 (50,0% / 50,0%) | Notebook, celda 8 |
| Train final / validación / test | 20.000 / 5.000 / 25.000 | Notebook, celdas 15 y 25 |
| Clases train final | 10.000 negativas / 10.000 positivas | Notebook, celda 25 |
| Clases validación | 2.500 negativas / 2.500 positivas | Notebook, celda 25 |
| Longitud tokens train final: media / mediana / p90 / p95 / p99 / máximo | 230,0 / 172 / 451,0 / 585,0 / 907,0 / 2.459 | Notebook, celda 17 |
| `MAX_LEN` elegido / reseñas truncadas | 585 / 4,96% | Notebook, celda 17 |
| `MAX_TOKENS` / vocabulario real | 20.000 / 20.000 | Notebook, celdas 19 y 29 |
| `SEED` / `BATCH_SIZE` | 42 / 64 | Notebook, celdas 3 y 29 |
| `% [UNK]` train final / validación / test | 3,291% / 3,638% / 4,043% | Notebook, celdas 22 y 29 |
| Hiperparámetros compartidos | 10 épocas; Adam; learning rate 0,001; `binary_crossentropy`; `accuracy`; embedding 32; Dense 16 ReLU; vocabulario 20.000; secuencia 585; semilla 42; batch 64 | Notebook, celdas 32 y 41 |
| Cambio exclusivo del mejorado | Dropout 0,5 | Notebook, celda 41 |
| Parámetros modelo base | 640.545 totales; 640.545 entrenables; 0 no entrenables; 2,44 MB | Notebook, celda 34 |
| Parámetros modelo mejorado | 640.545 totales; 640.545 entrenables; 0 no entrenables; 2,44 MB | Notebook, celda 43 |
| Base, época 1: accuracy / loss / val_accuracy / val_loss / tiempo | 0,7696 / 0,5495 / 0,8634 / 0,3700 / 1 s, 4 ms/step | Notebook, celda 36 |
| Base, época 2: accuracy / loss / val_accuracy / val_loss / tiempo | 0,8935 / 0,2878 / 0,8888 / 0,2797 / 1 s, 2 ms/step | Notebook, celda 36 |
| Base, época 3: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9255 / 0,2049 / 0,8982 / 0,2603 / 1 s, 2 ms/step | Notebook, celda 36 |
| Base, época 4: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9481 / 0,1569 / 0,9008 / 0,2585 / 1 s, 2 ms/step | Notebook, celda 36 |
| Base, época 5: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9628 / 0,1214 / 0,9020 / 0,2654 / 1 s, 3 ms/step | Notebook, celda 36 |
| Base, época 6: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9737 / 0,0933 / 0,9010 / 0,2785 / 1 s, 3 ms/step | Notebook, celda 36 |
| Base, época 7: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9833 / 0,0705 / 0,9000 / 0,2972 / 1 s, 3 ms/step | Notebook, celda 36 |
| Base, época 8: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9896 / 0,0523 / 0,8992 / 0,3199 / 1 s, 3 ms/step | Notebook, celda 36 |
| Base, época 9: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9936 / 0,0381 / 0,8940 / 0,3453 / 1 s, 3 ms/step | Notebook, celda 36 |
| Base, época 10: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9968 / 0,0274 / 0,8914 / 0,3734 / 1 s, 3 ms/step | Notebook, celda 36 |
| Mejorado, época 1: accuracy / loss / val_accuracy / val_loss / tiempo | 0,6977 / 0,6097 / 0,8570 / 0,4340 / 1 s, 4 ms/step | Notebook, celda 45 |
| Mejorado, época 2: accuracy / loss / val_accuracy / val_loss / tiempo | 0,8651 / 0,3608 / 0,8834 / 0,2952 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 3: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9059 / 0,2624 / 0,8940 / 0,2665 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 4: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9277 / 0,2130 / 0,8992 / 0,2584 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 5: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9442 / 0,1767 / 0,9016 / 0,2611 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 6: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9557 / 0,1490 / 0,9022 / 0,2699 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 7: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9644 / 0,1266 / 0,9028 / 0,2802 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 8: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9721 / 0,1050 / 0,8998 / 0,2970 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 9: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9776 / 0,0864 / 0,8942 / 0,3233 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejorado, época 10: accuracy / loss / val_accuracy / val_loss / tiempo | 0,9807 / 0,0770 / 0,8944 / 0,3418 / 1 s, 2 ms/step | Notebook, celda 45 |
| Mejor época base | Época 4; `val_loss=0,2585`; `val_accuracy=0,9008` | Notebook, celda 38 |
| Mejor época mejorado | Época 4; `val_loss=0,2584`; `val_accuracy=0,8992` | Notebook, celda 47 |
| Base: test loss / test accuracy | 0,4638 / 0,8596 | Notebook, celda 51 |
| Mejorado: test loss / test accuracy | 0,4198 / 0,8650 | Notebook, celda 51 |
| Base: brecha accuracy train-validación / aumento de val_loss | 0,1054 / 0,1149 | Notebook, celda 53 |
| Mejorado: brecha accuracy train-validación / aumento de val_loss | 0,0863 / 0,0835 | Notebook, celda 53 |
| Textos exactos repetidos train final ∩ test | 99 | Notebook, celdas 26, 53 y 56 |
| Mejorado: errores / falsos positivos / falsos negativos | 3.375 / 1.351 / 2.024 | Notebook, celda 55 |
| Matriz base `[[TN, FP], [FN, TP]]` | `[[11010, 1490], [2020, 10480]]` | Notebook, celda 64 |
| Matriz mejorado `[[TN, FP], [FN, TP]]` | `[[11149, 1351], [2024, 10476]]` | Notebook, celda 64 |
| Fallo A: `idx_test` / real / predicción | 5.220 / 0 (negativo) / 1 (positivo) | Notebook, celda 58 |
| Fallo A: probabilidad positiva / confianza / palabras | 1,000000 / 1,000000 / 4 | Notebook, celda 58 |
| Fallo A: texto literal | `Primary plot!Primary direction!Poor interpretation.` | Notebook, celda 58 |
| Caso B: predicción / confianza / baja confianza / umbral | negativo / 0,5835 / `true` / 0,70 | `assets/apuntes/fallo_real.md` (no aparece en salida del notebook) |
| Exportación `.keras`: tamaño | 7,58 MB | Notebook, celda 62 |
| Verificación ida y vuelta: diferencia absoluta máxima | 0,0 | Notebook, celda 63 |
