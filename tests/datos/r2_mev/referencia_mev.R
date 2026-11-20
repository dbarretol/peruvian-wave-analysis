# Referencia externa de R2 (contrato de la F7, §8): mev::NC.diag de Northrop y Coleman sobre dos conjuntos fijos.
# Se corre UNA vez, fuera de las pruebas; los números quedan en referencia_mev.csv y las pruebas los leen sin R.
# Uso (desde esta carpeta):  Rscript referencia_mev.R   (corrido con R 4.6.2 y mev 2.2)
library(mev)
options(digits = 17)
cat("mev", as.character(packageVersion("mev")), "\n")
filas <- list()
for (i in 1:2) {
  x <- read.csv(sprintf("conjunto_%d.csv", i))$x
  u <- read.csv(sprintf("conjunto_%d_umbrales.csv", i))$u
  res <- NC.diag(x, u, GP.fit = "Grimshaw", plot = FALSE)
  cat("conjunto", i, "\n"); str(res)
  # mev 2.2: lista con e.test.stats (S), df (m − i) y e.p.values (p) por umbral v_1, …, v_(m−1)
  filas[[i]] <- data.frame(conjunto = i, i_umbral = seq_along(res$e.p.values), gl = res$df,
                           s_score = res$e.test.stats, p_score = res$e.p.values)
}
write.csv(do.call(rbind, filas), "referencia_mev.csv", row.names = FALSE)
# 17 cifras para que la comparación no dependa del redondeo de write.csv
cat("Escrito referencia_mev.csv\n")
