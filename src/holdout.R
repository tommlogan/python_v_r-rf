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
library(parallel)

DATA_PATH = 'data/data.csv'
MODEL_NAME_PATH = 'data/predictions/model_names.csv'
HOLDOUT_NUM = 10
SEED = 15
set.seed(SEED)
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = TRUE
RESPONSE_VAR = 'y'

# MODEL <- 'r_rf_pyParams'
MODEL <- 'r_rf_default'

main <- function(){

  # import and clean data
  data <- ImportData()

  # holdout
  time.elapsed <- Cross.Validation(data)
  
  # append time
  write.table(c('R', MODEL, time.elapsed), 'data/time_elapsed.csv', append=T)
  
  # train and save model
  rf <- randomForst(y~.,data=data, ntree=10, mtry=1.0)
  saveRDS(rf, paste0('data/models/',MODEL,'.rds'))
}


ImportData <- function(){
  # Import processed data ---------------------------------------------------
  data <- read.csv(DATA_PATH)

  # ignore some variables
  data <- data[,-which(names(data) %in% c('x6','x7','x9','x10'))]
  # Factor variables
  data[,c('x48', 'x49','x50','x51','x52','x53','x54')] <- lapply(data[, c('x48', 'x49','x50','x51','x52','x53','x54')], as.factor)

  return(data)
}


Cross.Validation <- function(data){
  # Initializing ------------------------------------------------------------
  
  # Import the data divisions
  train.indices <- read.csv('data/holdout_indices.csv')
  train.indices <- train.indices[,-1]

  # create cluster
  cl <- makeCluster(CORES_NUM, type='SOCK', outfile ="")
  clusterExport(cl, c("randomForest","fwrite",'data','train.indices','Model.Cross.Validation','MODEL'), envir = environment())
  # conduct the parallelisation
  tic()
  pblapply(cl = cl, seq(1,HOLDOUT_NUM),function(j) Model.Cross.Validation(j))
  t <- toc()
  stopCluster(cl)
  return(t$toc-t$tic)
}


Model.Cross.Validation <- function(i){
  # Holdout replications ----------------------------------------------------

  # subset the data
  train <- data[train.indices[,i]=='True',]
  test <- data[train.indices[,i]=='False',]
  y.test <- test$y
  test$y <- NULL

  # create a matrix of predictions to save
  results <- data.frame(actual = y.test)

  # train the models
  # https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
  # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  rf <- randomForest(y~.,data=train)
  # rf <- randomForest(y~.,data=train, ntree=10, mtry=1.0)

  # predictions
  yhat <- invisible(predict(rf, newdata = test, type= "response"))
  results['rf'] <- yhat

  # save predictions to file
  filename.save <- paste0('data/predictions/', MODEL, '_r_', i, '.csv')
  fwrite(results, file = filename.save)
}
