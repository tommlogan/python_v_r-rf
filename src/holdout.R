# packages and libraries --------------------------------------------------
require(gam)
require(tree)
require(earth)
options(java.parameters = "-Xmx16g")
require(bartMachine)
require(randomForest)
require(gbm)
library(pbapply)
library(data.table)
library(tictoc)

# arbitrary seed so results are the same
set.seed(104)


kNumberCores <- 18
hurricane <- 'Matthew'
kDataPath <- paste0("data/processed/processed_data_Jacksonville_", hurricane, '.txt')
kDataName <- "data/processed/route_names.txt"

main <- function(){

  # import and clean data
  data <- ImportData()
  data.lm <- data$data
  routes <- data$routes

  # holdout
  Holdouts(data.lm, routes)

  # calculate the performance of the models
  performance.metrics <- CalculatePerformance(routes)

  # plot results
  PlotPerformance(performance.metrics)

}


ImportData <- function(){
  # Import processed data ---------------------------------------------------
  data.name <- kDataPath
  data <- read.csv(data.name)
  data.name <- kDataName
  routes <- read.csv(data.name)

  # (1) List of routes
  routes = routes[c(1:8,27:36),] ###############################################
  #routes = routes[c(1:5),]
  routes = as.character(routes)

  # (2) Keep data on routes leaving Jacksonville
  data = data[data$route %in% routes,]

  # (3) Factor variables
  data$route = factor(data$route)
  data$hurr = factor(data$hurr)
  data$ff = factor(data$ff)
  data$evac = factor(data$evac)
  data$wk_day = factor(data$wk_day)

  ####################### Reduce dataset for short run #################################3
  #data = data[data$landfall>=-200 & data$landfall<=200,]

  # (4) Create dataset for linear models
  data.lm = data
  data.lm$date <- NULL
  data.lm$windDir <- NULL
  data.lm$weatherDescript <- NULL
  data.lm$skyCond_code1 <- NULL
  data.lm$skyCond_code2 <- NULL
  data.lm$skyCond_code3 <- NULL
  data.lm$skyCond_num1 <- NULL
  data.lm$skyCond_num2 <- NULL
  data.lm$skyCond_num3 <- NULL
  data.lm$pressureSeaLevel_mb <- NULL
  data.lm$traveltime[is.infinite(data.lm$traveltime)]<-NA
  data.lm = na.omit(data.lm)

  return(list(data = data.lm, routes = routes))
}


Holdouts <- function(data.lm, routes){
  # Initializing ------------------------------------------------------------

  #Number of holdouts
  route.number <- length(routes)

  # conduct the parallelisation
  pblapply(seq(1,route.number),function(j) TrainPredict(j, routes, data.lm), cl = as.integer(kNumberCores))
}


TrainPredict <- function(i, routes, data.lm){
  # Holdout replications ----------------------------------------------------

  # route name
  route.name <- routes[i]

  # subset the data
  x.lm = which(data.lm$route != route.name)
  x.lm = c(x.lm,which(data.lm$route == route.name & data.lm$landfall > 0))
  train <- data.lm[x.lm,]

  y.lm <- which(data.lm$route == route.name & data.lm$landfall <= 0)
  test <- data.lm[y.lm,]
  y.test <- data.lm$traveltime[y.lm]

  # create a matrix of predictions to save
  results <- data.frame(actual = y.test)

  # train the models
  lm = invisible(lm(traveltime~.,data=train))

  gam = invisible(gam(traveltime~.+s(day)+s(hour)+s(minute)+s(windSpeed_kph)+s(tempAir_c)+s(tempDwpt_c)+s(relHumidity),data=train))

  cart= invisible(tree(traveltime~.,data=train))

  # bart = invisible(bartMachine(data.lm[x.lm, !names(data.lm)=="traveltime"],data.lm$traveltime[x.lm]))

  mars= invisible(earth(traveltime~., data=data.lm[x.lm,]))

  rf= randomForest(traveltime~.,data=train)

  # boost= invisible(gbm(traveltime~.,data=data.lm[x.lm,], distribution='gaussian',n.trees=500,interaction.depth =4))

  # nnet = invisible(nnet(traveltime~.,data=data.lm[x.lm,], size=10, linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100))

  # predictions
  yhat = invisible(rep(0, length(y.lm)))
  results['null'] = yhat

  yhat = invisible(predict(lm,newdata = test, type = "response"))
  results['lm'] = yhat

  yhat = invisible(predict(gam,newdata = data.lm[y.lm,], type = "response"))
  results['gam'] = yhat

  yhat = invisible(predict(cart,newdata = data.lm[y.lm,]))
  results['cart'] = yhat

  # yhat = invisible(predict(bart, new_data = data.lm[x.lm, !names(data.lm)=="traveltime"]))
  # results['bart'] = yhat
  # jgc()

  yhat = invisible(predict(mars, newdata = data.lm[y.lm,], type= "response"))
  results['mars'] = yhat

  yhat = invisible(predict(rf, newdata = data.lm[y.lm,], type= "response"))
  results['rf'] = yhat

  # yhat= invisible(predict(boost ,newdata = data.lm[y.lm,],n.trees =500))
  # results['boost'] = yhat

  # yhat = invisible(predict(nnet1, newdata = data.lm[y.lm,]))
  # results['nnet'] = yhat

  # save predictions to file
  filename.save <- paste0('data/results/results_', hurricane, '_', route.name, '.csv')
  fwrite(results, file = filename.save)
}


CalculatePerformance <- function(routes){
  # calculate the performance measures, based on the results
  # import the first data
  route.name <- routes[1]
  filename.results <- paste0('data/results/results_', hurricane, '_', route.name, '.csv')
  results <- fread(filename.results)

  # number of routes
  K <- length(routes)

  models.tested <- colnames(results)
  models.tested <- models.tested[models.tested != "actual"]

  # initialize tables for the performance measures
  # mean absolute error
  MAE = data.frame(matrix(0, ncol = length(models.tested), nrow = K))
  colnames(MAE) <- models.tested
  # mean square error
  MSE = data.frame(matrix(0, ncol = length(models.tested), nrow = K))
  colnames(MSE) <- models.tested

  # loop through the holdouts
  for (i in 1:K){
    # import the prediction data
    route.name <- routes[i]
    filename.results <- paste0('data/results/results_', hurricane, '_', route.name, '.csv')
    results <- fread(filename.results)

    # calculate MSE and MAE
    MAE[i,models.tested] <- lapply(models.tested, function(x){mean(abs(results[,..x][[1]]-results$actual))})
    MSE[i,models.tested] <- lapply(models.tested, function(x){mean((results[,..x][[1]]-results$actual)**2)})

  }

  # save and return
  performance.metrics = list(MAE = MAE, MSE = MSE)
  return(performance.metrics)


}


PlotPerformance <- function(performance.metrics){
  # extract data
  MSE = performance.metrics$MSE
  MAE = performance.metrics$MAE

  # means
  averages = data.frame(MSE = colMeans(MSE),MAE = colMeans(MAE))

  # plot MSE
  pdf(paste0("fig/results/", hurricane, "_ho_MSE_boxplot.pdf"))
  boxplot(MSE,main = "MSE",las=2)
  dev.off()

  # plot MAE
  pdf(paste0("fig/results/", hurricane, "_ho_MAE_boxplot.pdf"))
  boxplot(MAE,main = "MAE",las=2)
  dev.off()

  #Export metrics
  filename.save <- paste0("data/results/", hurricane, "_MSE.csv")
  fwrite(MSE, file = filename.save)

  filename.save <- paste0("data/results/", hurricane, "_MAE.csv")
  fwrite(MAE, file =filename.save)
}
