# Function for all data preprocessing
# Pass in a train or test set
library(plyr)
library(caret)
library(randomForest)

bnp.process <- function(data) {
    ID <- data$ID;
    if(!is.null(data$target)) {
        target <- data$target;
    }
    pca.list <- list(g8 = ~ v8 + v25 + v46 + v63 + v105,
                     g15 = ~ v15 + v32 + v73 + v86,
                     g17 = ~ v17 + v48 + v64 + v76,
                     g26 = ~ v26 + v43 + v60 + v116,
                     g29 = ~ v29 + v41 + v61 + v67 + v77 + v96,
                     g33 = ~ v33 + v55 + v83 + v111 + v121,
                     g34 = ~ v34 + v40 + v114,
                     g128 = ~ v108 + v109
                     );
    drop.list <- append(c('v107', 'v22', 'v110', 'v125', 'v56'),
                        unlist(llply(pca.list, function(f) attr(terms(f),'term.labels')),
                               use.names=F)
                        );
    data.drop <- data[,!(colnames(data) %in% drop.list)];
    data.pca <- data.drop;
    pca.cols <- llply(pca.list, function(l) { 
                      name <- paste('g', attr(terms(l),'term.labels')[1], sep='');
                      pca.model <- preProcess(model.frame(l, data, na.action=NULL), c('center', 'scale', 'pca'));
                      pca.res <- predict(pca.model, model.frame(l, data, na.action=NULL));
                      colnames(pca.res) <- lapply(colnames(pca.res),paste,name,sep='_');
                      return(cbind(ID, pca.res));
                     });
    l_ply(pca.cols, function(l) { data.pca <<- join(data.pca, l, by='ID')});
    data.cln <- as.data.frame(llply(data.pca, function(l) {
                                    if (is.factor(l)) {
                                        if (any(is.na(l))) {
                                            levels(l) <- append(levels(l),'');
                                            l[is.na(l)] <- "";
                                        }
                                        return(l);
                                    }
                                    l.mean <- mean(l, na.rm=T);
                                    l.std <- sd(l, na.rm=T);
                                    l[is.na(l)] <- rnorm(l[is.na(l)], mean=l.mean, sd=l.std);
                                    return(l);
                               }));

    if (is.null(data$target)) {
        data.cln$v71[data.cln$v71 %in% c('E', 'J', 'H')] <- 'F'
        data.cln$v71 <- factor(data.cln$v71)
        data.cln$v113[data.cln$v113 == 'K'] <- ""
        data.cln$v113 <- factor(data.cln$v113)
    }
    return(data.cln);
}

align.factor.levels <- function(train, test) {
    l_ply(colnames(test), function(n) { 
        if (is.factor(test[,n])) {
            levels(test[,n]) <- levels(train[,n])
        }
    })
    return(test)
}

kaggle.logloss <- function(actual, predicted) {
    predicted <- as.numeric(lapply(predicted, function(p) max(min(p,1-10^(-15)),10^(-15))));
    -(1/length(actual))*sum(actual*log(predicted)+(1-actual)*log(1-predicted));
}

set.seed(27)
train <- read.csv('train.csv')
test <- read.csv('test.csv')

train.cln <- bnp.process(train)
train.cln$target <- as.factor(train.cln$target)
levels(train.cln$target) <- c('high', 'low')
holdout.ind <- sample(1:nrow(train.cln), 0.1*nrow(train.cln),replace=F)
holdout <- train.cln[holdout.ind,]
train.cln.use <- train.cln[-holdout.ind,]
#Set up Cross-Validation
# cv.control <- trainControl(method='repeatedcv',
#                            number = 5,
#                            repeats= 2,
#                            classProbs=T,
#                            summaryFunction=multiClassSummary)
# set.seed(27)
# xgb.fit <- train(target ~ ., data=train.cln.use[,-1],
#                  method = "xgbTree",
#                  trControl = cv.control,
#                  verbose = T,
#                  metric = 'logLoss')
bnp.xgb.pred <- predict(xgb.fit, newdata=holdout, type='prob')

#bnp.rf <- randomForest(x=train.cln[,-c(1,2)], y=as.factor(train.cln$target), 
#                       ntree=500, nodesize=1, importance=T)
test.cln <- bnp.process(test)
#bnp.pred <- predict(bnp.rf, test.cln, type='prob')
bnp.pred <- predict(xgb.fit, newdata=test.cln, type='prob')
bnp.submit <- data.frame(ID=test.cln$ID, PredictedProb=bnp.pred[,'low'])