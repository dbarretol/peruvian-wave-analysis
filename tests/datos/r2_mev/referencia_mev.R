# Referencia externa de R2 (contrato de la F7, §8): mev::NC.diag de Northrop y Coleman sobre dos conjuntos fijos.
# Se corre UNA vez, fuera de las pruebas; los números quedan en referencia_mev.csv y las pruebas los leen sin R.
# Uso (desde esta carpeta):  Rscript referencia_mev.R
library(mev)
cat("mev", as.character(packageVersion("mev")), "\n")
filas <- list()
for (i in 1:2) {
  x <- read.csv(sprintf("conjunto_%d.csv", i))$x
  u <- read.csv(sprintf("conjunto_%d_umbrales.csv", i))$u
  res <- NC.diag(x, u, GP.fit = "Grimshaw", plot = FALSE)
  cat("conjunto", i, "\n"); str(res)
  # NC.diag devuelve (según la versión) un vector o una lista con los p de la prueba de score por umbral
  p <- if (is.list(res)) { if (!is.null(res$e.test)) res$e.test else unlist(res[[1]]) } else as.numeric(res)
  filas[[i]] <- data.frame(conjunto = i, i_umbral = seq_along(p), p_score = p)
}
write.csv(do.call(rbind, filas), "referencia_mev.csv", row.names = FALSE)
cat("Escrito referencia_mev.csv\n")
