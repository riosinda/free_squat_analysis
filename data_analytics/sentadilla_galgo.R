###Cargar libreria Galgo
library(galgo)
library(caret)
library(pROC)
library(e1071)

#Cargamos datos #De la base original se debe tranponer la parte de variables y generar otro csv
#Tambien se debe separar en otro archivo su clase
datos_sentadilla <- read.csv("C:/Users/guzma/OneDrive/MCPI/Proyectos/datos_sentadilla.csv", header=FALSE, row.names=1)
clase_sentadilla <- read.csv("C:/Users/guzma/OneDrive/MCPI/Proyectos/clase_sentadilla.csv", sep="", stringsAsFactors=TRUE)


datos_galgo_t=t(datos_sentadilla)
datos_galgo_t=cbind(datos_galgo_t,clase_sentadilla)

indices<-createDataPartition(datos_galgo_t$target,
                             times=1,
                             p=0.8, #proporcion de los datos 80% entrenamiento 
                             list=FALSE) #Indica que los índices se devolverán como un vector en lugar de una lista.
datosA<-datos_galgo_t[indices,] #contendrá el 80% de los datos originales, seleccionados aleatoriamente.
datosB<-datos_galgo_t[-indices,]
################################################################################
datosA_target <- datosA$target
datosA <- datosA[1:363]
datosA <- t(datosA)

#Primer paso configurar Galgo
bb.sentadilla = configBB.VarSel(
  data = datosA, #datos de busqueda
  classes = datosA_target, #Clases de los pacientes, FACTOR
  classification.method = "nearcent", 
  chromosomeSize = 5,
  maxSolutions = 726, #No de BB's #El doble de lo que se tenga mas 
  maxGenerations = 300, 
  goalFitness = 0.90, #alpha
  main = "Sentadilla",
  saveVariable = "bb.sentadilla",
  saveFrequency = 10,
  saveFile = "bb.sentadilla.rData"
)

blast(bb.sentadilla)

plot(bb.sentadilla, type="generankstability")
#plot(bb.sentadilla, type="confusion")

#USANDO LA LISTA ORDENADA DE MEJOR A PEOR CARACTERÍSTICA
#Crearemos un modelo usando la metodologia con la seleccion hacia delante
modelo_adelante = forwardSelectionModels(bb.sentadilla)

#El mejor es el #50 para mi caso
mejor_modelo = modelo_adelante$models[[5]]

#los nombres de las variables del mejor modelo
row.names(datos_sentadilla)[mejor_modelo]

#modelo reducido
#modelo_reducido=geneBackwardElimination(mejor_modelo,
                                        #bb.sentadilla,
                                        #result = "shortest")
formula <- "target ~ DOT_2_Euler_Y_max + 
  DOT_1_Euler_X_standard_deviation + 
  DOT_1_Euler_X_dinamic_range + 
  edad + 
  DOT_5_Euler_Z_mean + 
  DOT_2_Euler_Y_standard_deviation + 
  DOT_4_Euler_Y_variance + 
  DOT_2_Acc_X_dinamic_range + 
  DOT_5_Acc_Y_skewness + 
  DOT_4_Euler_Y_standard_deviation + 
  DOT_4_Euler_Y_dinamic_range + 
  DOT_1_Acc_X_min + 
  DOT_2_Euler_Y_mean + 
  DOT_2_Euler_Y_dinamic_range + 
  DOT_1_Acc_Z_standard_deviation + 
  DOT_1_Acc_Z_variance + 
  DOT_2_Acc_X_kurtosis + 
  DOT_1_Euler_Z_dinamic_range + 
  DOT_2_Euler_Z_min + 
  DOT_2_Euler_Y_min + 
  DOT_2_Euler_Z_mean + 
  DOT_1_Euler_Y_min + 
  DOT_2_Euler_Z_max + 
  DOT_1_Euler_Z_standard_deviation + 
  DOT_5_Euler_Z_max + 
  DOT_3_Euler_Y_dinamic_range"

datosA_t=datosA
datosA_t$target= datosA_target
modelo_final<-glm(formula, 
                data=datosA_t,
                family="binomial")

##############################################################################
summary(modelo_final)
predicciones_train <-predict(modelo_final, newdata = datosA_t,
                           type="response")
predicciones_test <- predict(modelo_final, newdata = datosB,
					type = "response")

tablita_train <- data.frame(Original = datosA_t$target,
				Predicciones = predicciones_train)

tablita_test <- data.frame(Original = datosB$target,
				Predicciones = predicciones_test)


roc_train <- roc(tablita_train$Original, tablita_train$Predicciones, 
			levels = c(0, 1), direction = 'auto')
roc_test <- roc(tablita_test$Original, tablita_test$Predicciones, 
			levels = c(0, 1), direction = 'auto')

plot(roc_train, col = "black", xlim = c(1,0))
lines(roc_test, col = "red")

legend("bottomright", legend = c("Train AUC: 0.9535", "Test AUC: 0.74"), col = c("black", "red"), lwd = 2)

###################################################################################

best_threshold_train <- coords(roc_train, "best", ret = "threshold")
best_threshold_test <- coords(roc_test, "best", ret = "threshold")

# Aplicar el mejor umbral a las predicciones
predicciones_train_bin <- ifelse(predicciones_train >= 0.5502575, 1, 0)
predicciones_test_bin <- ifelse(predicciones_test >= 0.5371942, 1, 0)

tablita_train_bin <- data.frame(Original = datosA_t$target, Predicciones = predicciones_train_bin)
tablita_test_bin <- data.frame(Original = datosB$target, Predicciones = predicciones_test_bin)

# Calcular las matrices de confusi?n
confusion_train <- confusionMatrix(reference = as.factor(tablita_train_bin$Original), data = as.factor(tablita_train_bin$Predicciones))
confusion_test <- confusionMatrix(reference = as.factor(tablita_test_bin$Original), data = as.factor(tablita_test_bin$Predicciones))

print(confusion_train)
print(confusion_test)

#######################################################################################

# Graficar las matrices de confusi?n
plot_confusion_matrix <- function(cm) {
  cm_table <- as.data.frame(cm$table)
  colnames(cm_table) <- c("Reference", "Prediction", "Freq")
  
  # Invertir la diagonal
  cm_table$Prediction <- factor(cm_table$Prediction, levels = rev(levels(cm_table$Prediction)))
  
  ggplot(data = cm_table, aes(x = Prediction, y = Reference)) +
    geom_tile(aes(fill = Freq), color = "white") +
    geom_text(aes(label = Freq), vjust = 1,size = 10) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Graficar la matriz de confusi?n para el conjunto de entrenamiento
plot_confusion_matrix(confusion_train)

# Graficar la matriz de confusi?n para el conjunto de prueba
plot_confusion_matrix(confusion_test)

# Mostrar m?tricas adicionales
metrics <- function(cm) {
  accuracy <- cm$overall['Accuracy']
  sensitivity <- cm$byClass['Sensitivity']
  specificity <- cm$byClass['Specificity']
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  
  list(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity,
       Precision = precision, Recall = recall)
}

metrics_train <- metrics(confusion_train)
metrics_test <- metrics(confusion_test)

print("M?tricas para Train:")
print(metrics_train)

print("M?tricas para Test:")
print(metrics_test)