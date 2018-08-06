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


HOLDOUT_NUM = 10
SEED = 15
set.seed(SEED)
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = TRUE
RESPONSE_VAR = 'y'

# MODEL <- 'r_rf_pyParams'
MODEL <- 'r_rf_default'
dataset <- 'lst'

setwd("C:/Users/tommc/OneDrive/Code/predictiveModelling/R_vs_Python/")

# Read model
rf <- readRDS(paste0('results/',dataset,'/models/r_rf_default.rds'))

# import data
DATA_PATH = paste0('data/data_',dataset,'.csv')
data <- read.csv(DATA_PATH)

if (DATA_PATH=='data/data_zeroinflate.csv'){
  # Factor variables
  data[,c('x48', 'x49','x50','x51','x52','x53','x54')] <- lapply(data[, c('x48', 'x49','x50','x51','x52','x53','x54')], as.factor)
}

# importance plot
varImpPlot(rf)

# partial dependence
imp = importance(rf)
imp_vars = rownames(imp)[order(imp,decreasing = T)[1:3]]
partialPlot(rf, x.var=imp_vars, pred.data=data)