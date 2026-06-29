# R reference analysis on identical benchmark data, for DroPhenix comparison.
# survival = the gold-standard R package (Therneau); PlantGrowth = base-R dataset.
suppressMessages(library(survival))

# ---- Freireich 1963 6-MP leukemia trial (weeks) ----
mp_t <- c(10,7,32,23,22,6,16,34,32,25,11,20,19,6,17,35,6,13,9,6,10)
mp_e <- c(1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0)
pl_t <- c(1,22,3,12,8,17,2,11,8,12,2,5,4,15,8,23,5,11,4,1,8)
pl_e <- rep(1,21)
time  <- c(mp_t, pl_t); event <- c(mp_e, pl_e)
grp   <- factor(c(rep("6MP",21), rep("Placebo",21)), levels=c("6MP","Placebo"))

sf  <- survfit(Surv(time,event) ~ grp)
med <- summary(sf)$table[, "median"]
sd  <- survdiff(Surv(time,event) ~ grp)                 # Mantel-Cox log-rank
chisq <- sd$chisq; p_lr <- pchisq(chisq, df=1, lower.tail=FALSE)
hr_breslow <- exp(coef(coxph(Surv(time,event) ~ grp, ties="breslow")))
hr_efron   <- exp(coef(coxph(Surv(time,event) ~ grp, ties="efron")))

cat(sprintf("MEDIAN_6MP=%.4f\n", med[[1]]))
cat(sprintf("MEDIAN_PLACEBO=%.4f\n", med[[2]]))
cat(sprintf("LOGRANK_CHISQ=%.5f\n", chisq))
cat(sprintf("LOGRANK_P=%.8f\n", p_lr))
cat(sprintf("COX_HR_BRESLOW=%.5f\n", hr_breslow))
cat(sprintf("COX_HR_EFRON=%.5f\n", hr_efron))

# ---- PlantGrowth (base R; Dobson 1983) ----
av <- summary(aov(weight ~ group, data = PlantGrowth))[[1]]
cat(sprintf("ANOVA_F=%.5f\n", av["group", "F value"]))
cat(sprintf("ANOVA_P=%.6f\n", av["group", "Pr(>F)"]))
kw <- kruskal.test(weight ~ group, data = PlantGrowth)
cat(sprintf("KRUSKAL_H=%.5f\n", as.numeric(kw$statistic)))
cat(sprintf("KRUSKAL_P=%.6f\n", kw$p.value))
