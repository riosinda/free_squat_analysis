library(caret)
library(ROCR)

data_freesquat <- read.csv("C:/Users/josei/OneDrive/Documentos/CIIBI proyectos/Analisys Free Squat/database/data_freesquat.csv")

# Select the interest variables (omitting the ID column)
datos <- data_freesquat[2:365]


#best 50 galgo features
#best_features <- c(
#  "edad", "DOT_5_Euler_Y_dinamic_range", "DOT_5_Euler_X_dinamic_range", "DOT_3_Acc_Z_max",
#  "DOT_4_Euler_Z_skewness", "DOT_3_Euler_Y_max", "DOT_3_Acc_X_variance", "DOT_5_Euler_Y_variance",
#  "DOT_4_Acc_Z_variance", "DOT_1_Acc_Z_standard_deviation", "DOT_5_Acc_Z_standard_deviation", "DOT_2_Gyr_Y_kurtosis",
#  "DOT_3_Acc_X_standard_deviation", "DOT_3_Euler_Y_mean", "peso", "DOT_3_Gyr_Y_max",
#  "DOT_1_Euler_Z_kurtosis", "DOT_1_Acc_X_skewness", "DOT_5_Euler_X_variance", "DOT_2_Acc_X_skewness",
#  "DOT_2_Acc_X_variance", "DOT_3_Euler_Y_dinamic_range", "DOT_4_Acc_Z_standard_deviation", "DOT_3_Acc_X_dinamic_range",
#  "DOT_1_Gyr_X_dinamic_range", "DOT_2_Gyr_X_kurtosis", "DOT_1_Euler_Y_kurtosis", "DOT_4_Euler_X_skewness",
#  "DOT_5_Acc_Y_skewness", "DOT_1_Gyr_Y_skewness", "DOT_5_Acc_Y_min", "DOT_1_Gyr_X_standard_deviation",
#  "DOT_5_Euler_Y_kurtosis", "DOT_2_Gyr_Y_skewness", "DOT_3_Gyr_Y_min", "DOT_3_Euler_X_mean",
#  "DOT_3_Gyr_Y_dinamic_range", "DOT_4_Euler_X_max", "DOT_4_Acc_X_dinamic_range", "DOT_1_Acc_Z_variance",
#  "DOT_3_Euler_Z_skewness", "DOT_5_Gyr_Y_max", "DOT_5_Euler_X_min", "DOT_1_Gyr_X_max",
#  "DOT_3_Acc_Y_max", "DOT_4_Gyr_Y_kurtosis", "DOT_1_Gyr_X_variance", "DOT_2_Acc_Z_variance",
#  "DOT_2_Acc_Z_max", "DOT_2_Acc_Y_dinamic_range"
#)

#best 12 galgo features
#best_features <- c(
#  "edad", "DOT_5_Euler_Y_dinamic_range", "DOT_5_Euler_X_dinamic_range", "DOT_3_Acc_Z_max",
#  "DOT_4_Euler_Z_skewness", "DOT_3_Euler_Y_max", "DOT_3_Acc_X_variance", "DOT_5_Euler_Y_variance",
#  "DOT_4_Acc_Z_variance", "DOT_1_Acc_Z_standard_deviation", "DOT_5_Acc_Z_standard_deviation", "DOT_2_Gyr_Y_kurtosis"
#)

#best 34 forward selecction features
best_features <- c(
  "peso", "edad", "DOT_1_Acc_Y_skewness", "DOT_1_Acc_Z_skewness", 
  "DOT_1_Acc_X_standard_deviation", "DOT_1_Acc_Z_dinamic_range", 
  "DOT_2_Euler_Y_mean", "DOT_2_Euler_X_variance", 
  "DOT_2_Euler_Y_variance", "DOT_2_Acc_X_variance", 
  "DOT_2_Acc_Y_variance", "DOT_2_Euler_Z_skewness", 
  "DOT_2_Acc_X_skewness", "DOT_2_Gyr_Y_skewness", 
  "DOT_2_Euler_X_standard_deviation", "DOT_2_Acc_X_standard_deviation", 
  "DOT_3_Euler_Z_mean", "DOT_3_Acc_Z_mean", 
  "DOT_3_Acc_Y_variance", "DOT_3_Gyr_X_standard_deviation", 
  "DOT_4_Euler_Y_variance", "DOT_4_Gyr_X_variance", 
  "DOT_4_Acc_X_skewness", "DOT_4_Acc_Z_skewness", 
  "DOT_4_Euler_X_kurtosis", "DOT_4_Euler_X_max", 
  "DOT_4_Euler_Y_max", "DOT_5_Euler_Z_variance", 
  "DOT_5_Acc_Y_variance", "DOT_5_Gyr_Z_variance", 
  "DOT_5_Acc_Y_kurtosis", "DOT_5_Euler_X_standard_deviation"
)

#formula auc
#formula <- "target ~ peso + edad + DOT_1_Euler_X_mean + DOT_1_Euler_Y_mean + DOT_1_Euler_Z_mean + DOT_1_Acc_X_mean + DOT_1_Acc_Y_mean + DOT_1_Acc_Z_mean + DOT_1_Gyr_X_mean + DOT_1_Gyr_Y_mean + DOT_1_Gyr_Z_mean + DOT_1_Euler_X_variance + DOT_1_Euler_Y_variance + DOT_1_Euler_Z_variance + DOT_1_Acc_X_variance + DOT_1_Acc_Y_variance + DOT_1_Acc_Z_variance + DOT_1_Gyr_Y_variance + DOT_1_Euler_Z_skewness + DOT_1_Gyr_X_skewness + DOT_1_Gyr_Z_skewness + DOT_1_Euler_X_kurtosis + DOT_1_Euler_Y_kurtosis + DOT_1_Acc_X_kurtosis + DOT_1_Acc_Y_kurtosis + DOT_1_Acc_Z_kurtosis + DOT_1_Gyr_X_kurtosis + DOT_1_Gyr_Y_kurtosis + DOT_1_Gyr_Z_kurtosis + DOT_1_Euler_X_standard_deviation + DOT_1_Euler_Y_standard_deviation + DOT_1_Euler_Z_standard_deviation + DOT_1_Acc_Y_standard_deviation + DOT_1_Acc_Z_standard_deviation + DOT_1_Gyr_X_standard_deviation + DOT_1_Gyr_Y_standard_deviation + DOT_1_Gyr_Z_standard_deviation + DOT_1_Euler_X_max + DOT_1_Euler_Z_max + DOT_1_Acc_X_max + DOT_1_Acc_Y_max + DOT_1_Gyr_Z_max + DOT_1_Euler_X_min + DOT_1_Euler_Y_min + DOT_1_Euler_Z_min + DOT_1_Acc_X_min + DOT_1_Acc_Y_min + DOT_1_Acc_Z_min + DOT_1_Gyr_Z_min + DOT_1_Euler_X_dinamic_range + DOT_1_Euler_Z_dinamic_range + DOT_1_Acc_X_dinamic_range + DOT_1_Acc_Y_dinamic_range + DOT_1_Acc_Z_dinamic_range + DOT_1_Gyr_Z_dinamic_range + DOT_2_Euler_X_mean + DOT_2_Euler_Y_mean + DOT_2_Euler_Z_mean + DOT_2_Acc_X_mean + DOT_2_Acc_Y_mean + DOT_2_Acc_Z_mean + DOT_2_Gyr_X_mean + DOT_2_Euler_X_variance + DOT_2_Euler_Y_variance + DOT_2_Euler_Z_variance + DOT_2_Acc_X_variance + DOT_2_Acc_Y_variance + DOT_2_Acc_Z_variance + DOT_2_Gyr_X_variance + DOT_2_Gyr_Y_variance + DOT_2_Gyr_Z_variance + DOT_2_Euler_X_skewness + DOT_2_Euler_Y_skewness + DOT_2_Euler_Z_skewness + DOT_2_Acc_X_skewness + DOT_2_Gyr_X_skewness + DOT_2_Gyr_Y_skewness + DOT_2_Euler_X_kurtosis + DOT_2_Acc_X_kurtosis + DOT_2_Acc_Y_kurtosis + DOT_2_Acc_Z_kurtosis + DOT_2_Gyr_X_kurtosis + DOT_2_Euler_X_standard_deviation + DOT_2_Euler_Y_standard_deviation + DOT_2_Acc_X_standard_deviation + DOT_2_Acc_Z_standard_deviation + DOT_2_Gyr_X_standard_deviation + DOT_2_Gyr_Y_standard_deviation + DOT_2_Euler_Z_max + DOT_2_Acc_X_max + DOT_2_Gyr_Y_max + DOT_2_Gyr_Z_max + DOT_2_Acc_Y_min + DOT_3_Euler_X_mean + DOT_3_Euler_Y_mean + DOT_3_Euler_Z_mean + DOT_3_Acc_Y_mean + DOT_3_Acc_Z_mean + DOT_3_Gyr_X_mean + DOT_3_Gyr_Z_mean + DOT_3_Euler_X_variance + DOT_3_Acc_Z_variance + DOT_3_Euler_Y_skewness + DOT_3_Acc_X_skewness + DOT_3_Acc_Y_skewness + DOT_3_Gyr_X_skewness + DOT_3_Gyr_Y_skewness + DOT_3_Gyr_Z_skewness + DOT_3_Euler_X_kurtosis + DOT_3_Acc_X_kurtosis + DOT_3_Acc_Y_kurtosis + DOT_3_Acc_Z_kurtosis + DOT_3_Gyr_X_kurtosis + DOT_3_Gyr_Z_kurtosis + DOT_3_Euler_X_standard_deviation + DOT_3_Euler_Y_standard_deviation + DOT_3_Euler_Z_standard_deviation + DOT_3_Acc_Z_standard_deviation + DOT_3_Gyr_Y_standard_deviation + DOT_3_Euler_X_dinamic_range + DOT_3_Euler_Z_dinamic_range + DOT_3_Acc_X_dinamic_range + DOT_3_Acc_Y_dinamic_range + DOT_3_Gyr_Y_dinamic_range + DOT_4_Euler_Y_mean + DOT_4_Euler_Z_mean + DOT_4_Acc_X_mean + DOT_4_Gyr_X_mean + DOT_4_Gyr_Y_mean + DOT_4_Acc_X_variance + DOT_4_Gyr_X_variance + DOT_4_Euler_X_skewness + DOT_4_Euler_Z_skewness + DOT_4_Acc_X_skewness + DOT_4_Acc_Y_skewness + DOT_4_Gyr_X_skewness + DOT_4_Gyr_Y_skewness + DOT_4_Gyr_Z_skewness + DOT_4_Euler_Y_kurtosis + DOT_4_Gyr_Z_standard_deviation + DOT_4_Gyr_Y_max + DOT_5_Euler_Z_mean + DOT_5_Acc_X_mean + DOT_5_Acc_Z_mean + DOT_5_Euler_Y_skewness + DOT_5_Euler_Z_skewness + DOT_5_Acc_Z_kurtosis + DOT_5_Euler_X_min"

#formula accuracy
#formula <- "target ~ peso + edad + DOT_1_Acc_Y_skewness + DOT_1_Acc_Z_skewness + DOT_1_Acc_X_standard_deviation + DOT_1_Acc_Z_dinamic_range + DOT_2_Euler_Y_mean + DOT_2_Euler_X_variance + DOT_2_Euler_Y_variance + DOT_2_Acc_X_variance + DOT_2_Acc_Y_variance + DOT_2_Euler_Z_skewness + DOT_2_Acc_X_skewness + DOT_2_Gyr_Y_skewness + DOT_2_Euler_X_standard_deviation + DOT_2_Acc_X_standard_deviation + DOT_3_Euler_Z_mean + DOT_3_Acc_Z_mean + DOT_3_Acc_Y_variance + DOT_3_Gyr_X_standard_deviation + DOT_4_Euler_Y_variance + DOT_4_Gyr_X_variance + DOT_4_Acc_X_skewness + DOT_4_Acc_Z_skewness + DOT_4_Euler_X_kurtosis + DOT_4_Euler_X_max + DOT_4_Euler_Y_max + DOT_5_Euler_Z_variance + DOT_5_Acc_Y_variance + DOT_5_Gyr_Z_variance + DOT_5_Acc_Y_kurtosis + DOT_5_Euler_X_standard_deviation"

formula_string <- paste("target", "~", paste(best_features, collapse = " + "))
formula <- as.formula(formula_string)

modelo = glm(formula, data = datos,
             family = "binomial")

summary(modelo)
predicciones = predict(modelo, newdata = datos, type = 'response')

predicciones_binarizadas = predicciones

tablita = data.frame(Original = datos$target,
                     Predicciones = predicciones_binarizadas)

#Paso 5(c-2) Calculamos la curva ROc para binarizar y saber el AUC
library(pROC)
mi_curva= roc(tablita$Original, tablita$Predicciones,
              levels=c(0,1), plot = TRUE, ci=TRUE,
              smooth=FALSE, direction='auto', col="blue",
              main="Mi curva chida")

mi_curva$auc #Calcular el area bajo la curva para ver que tan bueno es
mi_curva #Te da un intervalo de confianza 

#Obtenemos el umbral de corte
umbral = coords(mi_curva, "best", ret="threshold")

umbral <- as.numeric(umbral)
#Binarizo con el umbral obtenido
predicciones_binarizadas[predicciones_binarizadas < umbral] <- 0
predicciones_binarizadas[predicciones_binarizadas >= umbral] <-1


tablita2 = data.frame(Original = datos$target,
                      Predicciones = predicciones_binarizadas)


confusionMatrix(data = as.factor(tablita2$Predicciones),
                reference = as.factor(tablita2$Original),
                positive = '1')
