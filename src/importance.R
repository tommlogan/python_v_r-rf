# packages and libraries --------------------------------------------------
require(randomForest)
library(pbapply)
library(data.table)
library(parallel)


HOLDOUT_NUM = 10
SEED = 15
set.seed(SEED)
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = TRUE
RESPONSE_VAR = 'y'

# MODEL <- 'r_rf_pyParams'
MODEL <- 'r_rf_default'
dataset <- 'concrete'

# setwd("C:/Users/tommc/OneDrive/Code/predictiveModelling/R_vs_Python/")
setwd('B:/OneDrive/Code/predictiveModelling/R_vs_Python')
DATA_PATH = paste0('data/data_',dataset,'.csv')
data <- read.csv(DATA_PATH, stringsAsFactors=FALSE)
# Read model
# rf <- readRDS(paste0('results/',dataset,'/models/r_rf_default.rds'))
rf <- randomForest(y~.,data=data)

# import data
DATA_PATH = paste0('data/data_',dataset,'.csv')
data <- read.csv(DATA_PATH)

if (DATA_PATH=='data/data_zeroinflate.csv'){
  # Factor variables
  data[,c('data', 'x49','x50','x51','x52','x53','x54')] <- lapply(data[, c('x48', 'x49','x50','x51','x52','x53','x54')], as.factor)
} else if (DATA_PATH=='data/data_lst.csv') {
  # all numeric
  data <- as.data.frame(sapply( data, as.numeric ))
  data <- data[complete.cases(data),]
}

# importance plot
jpeg(paste0("fig/varimp_R_",dataset,".jpg"), width = 1280, height = 960, units = "px", pointsize=18)
varImpPlot(rf, type=2, main=paste0('R Feature Importance: ',dataset))

dev.off()


# partial dependence
imp = importance(rf)
imp_vars = rownames(imp)[order(imp,decreasing = T)[1:4]]

jpeg(paste0("fig/pdp_R_",dataset,".jpg"), width = 1280, height = 960, units = "px", pointsize=18)
op <- par(mfrow=c(2, 2))
for (i in seq_along(imp_vars)){
  partialPlot(rf, data, imp_vars[i], xlab=imp_vars[i],
                main='',
                ylim=c(30, 70))
}
dev.off()